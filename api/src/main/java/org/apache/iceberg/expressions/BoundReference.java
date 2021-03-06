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

package org.apache.iceberg.expressions;

import org.apache.iceberg.Accessor;
import org.apache.iceberg.StructLike;
import org.apache.iceberg.types.Type;

public class BoundReference<T> implements Reference {
  private final int fieldId;
  private final Accessor<StructLike> accessor;

  BoundReference(int fieldId, Accessor<StructLike> accessor) {
    this.fieldId = fieldId;
    this.accessor = accessor;
  }

  public Type type() {
    return accessor.type();
  }

  public int fieldId() {
    return fieldId;
  }

  public Accessor<StructLike> accessor() {
    return accessor;
  }

  @SuppressWarnings("unchecked")
  public T get(StructLike struct) {
    return (T) accessor.get(struct);
  }

  @Override
  public String toString() {
    return String.format("ref(id=%d, accessor-type=%s)", fieldId, accessor.type());
  }

}
