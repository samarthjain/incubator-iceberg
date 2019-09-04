/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.iceberg.parquet;

import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import java.io.Closeable;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.function.Function;
import org.apache.iceberg.Schema;
import org.apache.iceberg.exceptions.RuntimeIOException;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.expressions.Expressions;
import org.apache.iceberg.io.CloseableGroup;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.io.InputFile;
import org.apache.parquet.ParquetReadOptions;
import org.apache.parquet.column.page.PageReadStore;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.schema.MessageType;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.vectorized.ColumnarBatch;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static org.apache.iceberg.parquet.ParquetSchemaUtil.addFallbackIds;
import static org.apache.iceberg.parquet.ParquetSchemaUtil.hasIds;
import static org.apache.iceberg.parquet.ParquetSchemaUtil.pruneColumns;
import static org.apache.iceberg.parquet.ParquetSchemaUtil.pruneColumnsFallback;

public class ParquetReader<T> extends CloseableGroup implements CloseableIterable<T> {
  private final InputFile input;
  private final Schema expectedSchema;
  private final ParquetReadOptions options;
  private final Function<MessageType, BatchedReader> readerFunc;
  private final Expression filter;
  private final boolean reuseContainers;
  private final boolean caseSensitive;
  private static final Logger LOG = LoggerFactory.getLogger(ParquetReader.class);

  public ParquetReader(InputFile input, Schema expectedSchema, ParquetReadOptions options,
      Function<MessageType, BatchedReader> readerFunc,
      Expression filter, boolean reuseContainers, boolean caseSensitive,
      StructType sparkSchema, int maxRecordsPerBatch) {
    this.input = input;
    this.expectedSchema = expectedSchema;
    this.options = options;
    this.readerFunc = readerFunc;
    // replace alwaysTrue with null to avoid extra work evaluating a trivial filter
    this.filter = filter == Expressions.alwaysTrue() ? null : filter;
    this.reuseContainers = reuseContainers;
    this.caseSensitive = caseSensitive;
  }

  private static class ReadConf<T> {
    private final ParquetFileReader reader;
    private final InputFile file;
    private final ParquetReadOptions options;
    private final MessageType projection;
    private final ColumnarBatchReader model;
    private final List<BlockMetaData> rowGroups;
    private final boolean[] shouldSkip;
    private final long totalValues;
    private final boolean reuseContainers;

    @SuppressWarnings("unchecked")
    ReadConf(InputFile file, ParquetReadOptions options, Schema expectedSchema, Expression filter,
        Function<MessageType, ColumnarBatchReader> readerFunc, boolean reuseContainers,
        boolean caseSensitive) {
      this.file = file;
      this.options = options;
      this.reader = newReader(file, options);

      MessageType fileSchema = reader.getFileMetaData().getSchema();

      boolean hasIds = hasIds(fileSchema);
      MessageType typeWithIds = hasIds ? fileSchema : addFallbackIds(fileSchema);

      this.projection = hasIds ?
          pruneColumns(fileSchema, expectedSchema) :
          pruneColumnsFallback(fileSchema, expectedSchema);
      this.model = readerFunc.apply(typeWithIds);
      this.rowGroups = reader.getRowGroups();
      this.shouldSkip = new boolean[rowGroups.size()];

      ParquetMetricsRowGroupFilter statsFilter = null;
      ParquetDictionaryRowGroupFilter dictFilter = null;
      if (filter != null) {
        statsFilter = new ParquetMetricsRowGroupFilter(expectedSchema, filter, caseSensitive);
        dictFilter = new ParquetDictionaryRowGroupFilter(expectedSchema, filter, caseSensitive);
      }

      long totalValues = 0L;
      for (int i = 0; i < shouldSkip.length; i += 1) {
        BlockMetaData rowGroup = rowGroups.get(i);
        boolean shouldRead = filter == null || (
            statsFilter.shouldRead(typeWithIds, rowGroup) &&
                dictFilter.shouldRead(typeWithIds, rowGroup, reader.getDictionaryReader(rowGroup)));
        this.shouldSkip[i] = !shouldRead;
        if (shouldRead) {
          totalValues += rowGroup.getRowCount();
        }
      }

      this.totalValues = totalValues;
      this.reuseContainers = reuseContainers;
    }

    ReadConf(ReadConf<T> toCopy) {
      this.reader = null;
      this.file = toCopy.file;
      this.options = toCopy.options;
      this.projection = toCopy.projection;
      this.model = toCopy.model;
      this.rowGroups = toCopy.rowGroups;
      this.shouldSkip = toCopy.shouldSkip;
      this.totalValues = toCopy.totalValues;
      this.reuseContainers = toCopy.reuseContainers;
    }

    ParquetFileReader reader() {
      if (reader != null) {
        reader.setRequestedSchema(projection);
        return reader;
      }

      ParquetFileReader newReader = newReader(file, options);
      newReader.setRequestedSchema(projection);
      return newReader;
    }

    ColumnarBatchReader model() {
      return model;
    }

    boolean[] shouldSkip() {
      return shouldSkip;
    }

    long totalValues() {
      return totalValues;
    }

    boolean reuseContainers() {
      return reuseContainers;
    }

    ReadConf<T> copy() {
      return new ReadConf<>(this);
    }

    private static ParquetFileReader newReader(InputFile file, ParquetReadOptions options) {
      try {
        return ParquetFileReader.open(ParquetIO.file(file), options);
      } catch (IOException e) {
        throw new RuntimeIOException(e, "Failed to open Parquet file: %s", file.location());
      }
    }
  }

  private ReadConf conf = null;

  private ReadConf init() {
    if (conf == null) {
      ReadConf<T> conf = new ReadConf(
          input, options, expectedSchema, filter, readerFunc, reuseContainers, caseSensitive);
      this.conf = conf.copy();
      return conf;
    }

    return conf;
  }

  @Override
  public Iterator iterator() {
    // create iterator over file
    FileIterator iter = new FileIterator(init());
    addCloseable(iter);

    return iter;
  }

  private static class FileIterator implements Iterator, Closeable {
    private final ParquetFileReader reader;
    private final boolean[] shouldSkip;
    private final ColumnarBatchReader model;
    private final long totalValues;
    private final boolean reuseContainers;

    private int nextRowGroup = 0;
    private long nextRowGroupStart = 0;
    private long valuesRead = 0;
    private ColumnarBatch last = null;
    private final int totalRowGroups;
    // state effected when next row group is read is
    // model, nextRowGroup, nextRowGroupStart
    private static ExecutorService prefetchService =
        MoreExecutors.getExitingExecutorService((ThreadPoolExecutor) Executors.newFixedThreadPool(
            4,
            new ThreadFactoryBuilder()
                .setDaemon(true)
                .setNameFormat("iceberg-parquet-rowgroup-prefetchNext-pool-%d")
                .build()));

    // State associated with prefetching row groups
    private int prefetchedRowGroup;
    private Future<PageReadStore> prefetchRowGroupFuture;
    private long valuesReadFromThisRowGroup = 0l;
    private long totalValsInThisRowGroup = 0l;

    FileIterator(ReadConf conf) {
      this.reader = conf.reader();
      this.shouldSkip = conf.shouldSkip();
      this.model = conf.model();
      this.totalValues = conf.totalValues();
      this.reuseContainers = conf.reuseContainers();
      this.totalRowGroups = conf.rowGroups.size();
      prefetchNextRowGroup();
      advance();
    }

    @Override
    public boolean hasNext() {
      return valuesRead < totalValues;
    }

    @Override
    public ColumnarBatch next() {
      if (!hasNext()) {
        throw new NoSuchElementException("No more row groups to read");
      }
      if (valuesRead >= nextRowGroupStart) {
        advance();
      }
      if (reuseContainers) {
        this.last = model.read(null); // anjali-todo last was being reused here?
      } else {
        this.last = model.read(null);
      }
      valuesRead += last.numRows();
      valuesReadFromThisRowGroup += last.numRows();
      return last;
    }

    private void advance() {
      try {
        checkNotNull(prefetchRowGroupFuture);
        PageReadStore pages = prefetchRowGroupFuture.get();
        checkState(pages != null, "advance() should have been only when there was at least one row group to read");
        nextRowGroupStart += pages.getRowCount();
        nextRowGroup = prefetchedRowGroup + 1;
        model.setPageSource(pages);
        prefetchNextRowGroup(); // eagerly fetch the next row group
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new RuntimeException(e);
      } catch (ExecutionException e) {
        throw new RuntimeException(e);
      }
    }

    private void prefetchNextRowGroup() {
      prefetchRowGroupFuture = prefetchService.submit(() -> {
        prefetchedRowGroup = nextRowGroup;
        while (prefetchedRowGroup < totalRowGroups && shouldSkip[prefetchedRowGroup]) {
          prefetchedRowGroup += 1;
          reader.skipNextRowGroup();
        }
        try {
          if (prefetchedRowGroup < totalRowGroups) {
            PageReadStore pageReadStore = reader.readNextRowGroup();
            return pageReadStore;
          }
          return null;
        } catch (IOException e) {
          throw new RuntimeIOException(e);
        }
      });
    }


    @Override
    public void close() throws IOException {
      reader.close();
    }
  }
}
