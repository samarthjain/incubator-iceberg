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

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nullable;
import org.apache.iceberg.Schema;
import org.apache.iceberg.exceptions.RuntimeIOException;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.io.InputFile;
import org.apache.parquet.ParquetReadOptions;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.hadoop.metadata.ColumnChunkMetaData;
import org.apache.parquet.hadoop.metadata.ColumnPath;
import org.apache.parquet.schema.MessageType;

/**
 * Configuration for Parquet readers.
 *
 * @param <T> type of value to read
 */
class ReadConf<T> {
  private final ParquetFileReader reader;
  private final InputFile file;
  private final ParquetReadOptions options;
  private final MessageType projection;
  @Nullable
  private final ParquetValueReader<T> model;
  @Nullable
  private final VectorizedReader<T> vectorizedModel;
  private final List<BlockMetaData> rowGroups;
  private final boolean[] shouldSkip;
  private final long totalValues;
  private final boolean reuseContainers;
  @Nullable
  private final Integer batchSize;

  // Indexed by row group with nulls for row groups that are skipped
  private final List<Map<ColumnPath, ColumnChunkMetaData>> columnChunkMetaDataForRowGroups;

  @SuppressWarnings("unchecked")
  ReadConf(InputFile file, ParquetReadOptions options, Schema expectedSchema, Expression filter,
           Function<MessageType, ParquetValueReader<?>> readerFunc, Function<MessageType,
           VectorizedReader<?>> batchedReaderFunc, boolean reuseContainers,
           boolean caseSensitive, Integer bSize) {
    this.file = file;
    this.options = options;
    this.reader = newReader(file, options);
    MessageType fileSchema = reader.getFileMetaData().getSchema();

    boolean hasIds = ParquetSchemaUtil.hasIds(fileSchema);
    MessageType typeWithIds = hasIds ? fileSchema : ParquetSchemaUtil.addFallbackIds(fileSchema);

    this.projection = hasIds ?
        ParquetSchemaUtil.pruneColumns(fileSchema, expectedSchema) :
        ParquetSchemaUtil.pruneColumnsFallback(fileSchema, expectedSchema);
    this.rowGroups = reader.getRowGroups();
    this.shouldSkip = new boolean[rowGroups.size()];

    ParquetMetricsRowGroupFilter statsFilter = null;
    ParquetDictionaryRowGroupFilter dictFilter = null;
    if (filter != null) {
      statsFilter = new ParquetMetricsRowGroupFilter(expectedSchema, filter, caseSensitive);
      dictFilter = new ParquetDictionaryRowGroupFilter(expectedSchema, filter, caseSensitive);
    }

    long computedTotalValues = 0L;
    for (int i = 0; i < shouldSkip.length; i += 1) {
      BlockMetaData rowGroup = rowGroups.get(i);
      boolean shouldRead = filter == null || (
          statsFilter.shouldRead(typeWithIds, rowGroup) &&
              dictFilter.shouldRead(typeWithIds, rowGroup, reader.getDictionaryReader(rowGroup)));
      this.shouldSkip[i] = !shouldRead;
      if (shouldRead) {
        computedTotalValues += rowGroup.getRowCount();
      }
    }

    this.totalValues = computedTotalValues;
    if (readerFunc != null) {
      this.model = (ParquetValueReader<T>) readerFunc.apply(typeWithIds);
      this.vectorizedModel = null;
      this.columnChunkMetaDataForRowGroups = null;
    } else {
      this.model = null;
      this.vectorizedModel = (VectorizedReader<T>) batchedReaderFunc.apply(typeWithIds);
      this.columnChunkMetaDataForRowGroups =
          Stream.generate((Supplier<Map<ColumnPath, ColumnChunkMetaData>>) () -> null)
              .limit(rowGroups.size()).collect(Collectors.toList());
      populateColumnChunkMetadataForRowGroups();
    }
    this.reuseContainers = reuseContainers;
    this.batchSize = bSize;
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
    this.batchSize = toCopy.batchSize;
    this.vectorizedModel = toCopy.vectorizedModel;
    this.columnChunkMetaDataForRowGroups = toCopy.columnChunkMetaDataForRowGroups;
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

  ParquetValueReader<T> model() {
    return model;
  }

  VectorizedReader<T> vectorizedModel() {
    return vectorizedModel;
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

  Integer batchSize() {
    return batchSize;
  }

  List<Map<ColumnPath, ColumnChunkMetaData>> columnChunkMetadataForRowGroups() {
    return Collections.unmodifiableList(columnChunkMetaDataForRowGroups);
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

  private void populateColumnChunkMetadataForRowGroups() {
    Set<ColumnPath> projectedColumns = projection.getColumns().stream()
        .map(columnDescriptor -> ColumnPath.get(columnDescriptor.getPath())).collect(Collectors.toSet());
    for (int i = 0; i < rowGroups.size(); i++) {
      if (!shouldSkip[i]) {
        BlockMetaData blockMetaData = rowGroups.get(i);
        Map<ColumnPath, ColumnChunkMetaData> map = new HashMap<>();
        blockMetaData.getColumns().stream()
            .filter(columnChunkMetaData -> projectedColumns.contains(columnChunkMetaData.getPath()))
            .forEach(columnChunkMetaData -> map.put(columnChunkMetaData.getPath(), columnChunkMetaData));
        columnChunkMetaDataForRowGroups.set(i, map);
      }
    }
  }
}