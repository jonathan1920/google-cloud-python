# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import collections
import json

try:
    import fastavro
except ImportError:  # pragma: NO COVER
    fastavro = None
import google.api_core.exceptions

try:
    import pandas
except ImportError:  # pragma: NO COVER
    pandas = None
import six

from google.cloud.bigquery_storage_v1beta1 import types


_STREAM_RESUMPTION_EXCEPTIONS = (google.api_core.exceptions.ServiceUnavailable,)
_FASTAVRO_REQUIRED = (
    "fastavro is required to parse ReadRowResponse messages with Avro bytes."
)
_PANDAS_REQUIRED = "pandas is required to create a DataFrame"


class ReadRowsStream(object):
    """A stream of results from a read rows request.

    This stream is an iterable of
    :class:`~google.cloud.bigquery_storage_v1beta1.types.ReadRowsResponse`.
    Iterate over it to fetch all row messages.

    If the fastavro library is installed, use the
    :func:`~google.cloud.bigquery_storage_v1beta1.reader.ReadRowsStream.rows()`
    method to parse all messages into a stream of row dictionaries.

    If the pandas and fastavro libraries are installed, use the
    :func:`~google.cloud.bigquery_storage_v1beta1.reader.ReadRowsStream.to_dataframe()`
    method to parse all messages into a :class:`pandas.DataFrame`.
    """

    def __init__(self, wrapped, client, read_position, read_rows_kwargs):
        """Construct a ReadRowsStream.

        Args:
            wrapped (Iterable[ \
                ~google.cloud.bigquery_storage_v1beta1.types.ReadRowsResponse \
            ]):
                The ReadRows stream to read.
            client ( \
                ~google.cloud.bigquery_storage_v1beta1.gapic. \
                    big_query_storage_client.BigQueryStorageClient \
            ):
                A GAPIC client used to reconnect to a ReadRows stream. This
                must be the GAPIC client to avoid a circular dependency on
                this class.
            read_position (Union[ \
                dict, \
                ~google.cloud.bigquery_storage_v1beta1.types.StreamPosition \
            ]):
                Required. Identifier of the position in the stream to start
                reading from. The offset requested must be less than the last
                row read from ReadRows. Requesting a larger offset is
                undefined. If a dict is provided, it must be of the same form
                as the protobuf message
                :class:`~google.cloud.bigquery_storage_v1beta1.types.StreamPosition`
            read_rows_kwargs (dict):
                Keyword arguments to use when reconnecting to a ReadRows
                stream.

        Returns:
            Iterable[ \
                ~google.cloud.bigquery_storage_v1beta1.types.ReadRowsResponse \
            ]:
                A sequence of row messages.
        """

        # Make a copy of the read position so that we can update it without
        # mutating the original input.
        self._position = _copy_stream_position(read_position)
        self._client = client
        self._wrapped = wrapped
        self._read_rows_kwargs = read_rows_kwargs

    def __iter__(self):
        """An iterable of messages.

        Returns:
            Iterable[ \
                ~google.cloud.bigquery_storage_v1beta1.types.ReadRowsResponse \
            ]:
                A sequence of row messages.
        """

        # Infinite loop to reconnect on reconnectable errors while processing
        # the row stream.
        while True:
            try:
                for message in self._wrapped:
                    rowcount = message.avro_rows.row_count
                    self._position.offset += rowcount
                    yield message

                return  # Made it through the whole stream.
            except _STREAM_RESUMPTION_EXCEPTIONS:
                # Transient error, so reconnect to the stream.
                pass

            self._reconnect()

    def _reconnect(self):
        """Reconnect to the ReadRows stream using the most recent offset."""
        self._wrapped = self._client.read_rows(
            _copy_stream_position(self._position), **self._read_rows_kwargs
        )

    def rows(self, read_session):
        """Iterate over all rows in the stream.

        This method requires the fastavro library in order to parse row
        messages.

        .. warning::
            DATETIME columns are not supported. They are currently parsed as
            strings in the fastavro library.

        Args:
            read_session ( \
                ~google.cloud.bigquery_storage_v1beta1.types.ReadSession \
            ):
                The read session associated with this read rows stream. This
                contains the schema, which is required to parse the data
                messages.

        Returns:
            Iterable[Mapping]:
                A sequence of rows, represented as dictionaries.
        """
        if fastavro is None:
            raise ImportError(_FASTAVRO_REQUIRED)

        return ReadRowsIterable(self, read_session)

    def to_dataframe(self, read_session, dtypes=None):
        """Create a :class:`pandas.DataFrame` of all rows in the stream.

        This method requires the pandas libary to create a data frame and the
        fastavro library to parse row messages.

        .. warning::
            DATETIME columns are not supported. They are currently parsed as
            strings.

        Args:
            read_session ( \
                ~google.cloud.bigquery_storage_v1beta1.types.ReadSession \
            ):
                The read session associated with this read rows stream. This
                contains the schema, which is required to parse the data
                messages.
            dtypes ( \
                Map[str, Union[str, pandas.Series.dtype]] \
            ):
                Optional. A dictionary of column names pandas ``dtype``s. The
                provided ``dtype`` is used when constructing the series for
                the column specified. Otherwise, the default pandas behavior
                is used.

        Returns:
            pandas.DataFrame:
                A data frame of all rows in the stream.
        """
        if fastavro is None:
            raise ImportError(_FASTAVRO_REQUIRED)
        if pandas is None:
            raise ImportError(_PANDAS_REQUIRED)

        return self.rows(read_session).to_dataframe(dtypes=dtypes)


class ReadRowsIterable(object):
    """An iterable of rows from a read session.

    Args:
        reader (google.cloud.bigquery_storage_v1beta1.reader.ReadRowsStream):
            A read rows stream.
        read_session (google.cloud.bigquery_storage_v1beta1.types.ReadSession):
            A read session. This is required because it contains the schema
            used in the stream messages.
    """

    # This class is modelled after the google.cloud.bigquery.table.RowIterator
    # and aims to be API compatible where possible.

    def __init__(self, reader, read_session):
        self._status = None
        self._reader = reader
        self._read_session = read_session

    @property
    def total_rows(self):
        """int: Number of estimated rows in the current stream.

        May change over time.
        """
        return getattr(self._status, "estimated_row_count", None)

    @property
    def pages(self):
        """A generator of all pages in the stream.

        Returns:
            types.GeneratorType[google.cloud.bigquery_storage_v1beta1.ReadRowsPage]:
                A generator of pages.
        """
        # Each page is an iterator of rows. But also has num_items, remaining,
        # and to_dataframe.
        stream_parser = _StreamParser(self._read_session)
        for message in self._reader:
            self._status = message.status
            yield ReadRowsPage(stream_parser, message)

    def __iter__(self):
        """Iterator for each row in all pages."""
        for page in self.pages:
            for row in page:
                yield row

    def to_dataframe(self, dtypes=None):
        """Create a :class:`pandas.DataFrame` of all rows in the stream.

        This method requires the pandas libary to create a data frame and the
        fastavro library to parse row messages.

        .. warning::
            DATETIME columns are not supported. They are currently parsed as
            strings in the fastavro library.

        Args:
            dtypes ( \
                Map[str, Union[str, pandas.Series.dtype]] \
            ):
                Optional. A dictionary of column names pandas ``dtype``s. The
                provided ``dtype`` is used when constructing the series for
                the column specified. Otherwise, the default pandas behavior
                is used.

        Returns:
            pandas.DataFrame:
                A data frame of all rows in the stream.
        """
        if pandas is None:
            raise ImportError(_PANDAS_REQUIRED)

        frames = []
        for page in self.pages:
            frames.append(page.to_dataframe(dtypes=dtypes))
        return pandas.concat(frames)


class ReadRowsPage(object):
    """An iterator of rows from a read session message.

    Args:
        stream_parser (google.cloud.bigquery_storage_v1beta1.reader._StreamParser):
            A helper for parsing messages into rows.
        message (google.cloud.bigquery_storage_v1beta1.types.ReadRowsResponse):
            A message of data from a read rows stream.
    """

    # This class is modeled after google.api_core.page_iterator.Page and aims
    # to provide API compatibility where possible.

    def __init__(self, stream_parser, message):
        self._stream_parser = stream_parser
        self._message = message
        self._iter_rows = None
        self._num_items = self._message.avro_rows.row_count
        self._remaining = self._message.avro_rows.row_count

    def _parse_rows(self):
        """Parse rows from the message only once."""
        if self._iter_rows is not None:
            return

        rows = self._stream_parser.to_rows(self._message)
        self._iter_rows = iter(rows)

    @property
    def num_items(self):
        """int: Total items in the page."""
        return self._num_items

    @property
    def remaining(self):
        """int: Remaining items in the page."""
        return self._remaining

    def __iter__(self):
        """A ``ReadRowsPage`` is an iterator."""
        return self

    def next(self):
        """Get the next row in the page."""
        self._parse_rows()
        if self._remaining > 0:
            self._remaining -= 1
        return six.next(self._iter_rows)

    # Alias needed for Python 2/3 support.
    __next__ = next

    def to_dataframe(self, dtypes=None):
        """Create a :class:`pandas.DataFrame` of rows in the page.

        This method requires the pandas libary to create a data frame and the
        fastavro library to parse row messages.

        .. warning::
            DATETIME columns are not supported. They are currently parsed as
            strings in the fastavro library.

        Args:
            dtypes ( \
                Map[str, Union[str, pandas.Series.dtype]] \
            ):
                Optional. A dictionary of column names pandas ``dtype``s. The
                provided ``dtype`` is used when constructing the series for
                the column specified. Otherwise, the default pandas behavior
                is used.

        Returns:
            pandas.DataFrame:
                A data frame of all rows in the stream.
        """
        if pandas is None:
            raise ImportError(_PANDAS_REQUIRED)
        return self._stream_parser.to_dataframe(self._message, dtypes=dtypes)


class _StreamParser(object):
    """Helper to parse Avro messages into useful representations."""

    def __init__(self, read_session):
        """Construct a _StreamParser.

        Args:
            read_session (google.cloud.bigquery_storage_v1beta1.types.ReadSession):
                A read session. This is required because it contains the schema
                used in the stream messages.
        """
        self._read_session = read_session
        self._avro_schema_json = None
        self._fastavro_schema = None
        self._column_names = None

    def to_dataframe(self, message, dtypes=None):
        """Create a :class:`pandas.DataFrame` of rows in the page.

        This method requires the pandas libary to create a data frame and the
        fastavro library to parse row messages.

        .. warning::
            DATETIME columns are not supported. They are currently parsed as
            strings in the fastavro library.

        Args:
            dtypes ( \
                Map[str, Union[str, pandas.Series.dtype]] \
            ):
                Optional. A dictionary of column names pandas ``dtype``s. The
                provided ``dtype`` is used when constructing the series for
                the column specified. Otherwise, the default pandas behavior
                is used.

        Returns:
            pandas.DataFrame:
                A data frame of all rows in the stream.
        """
        self._parse_avro_schema()

        if dtypes is None:
            dtypes = {}

        columns = collections.defaultdict(list)
        for row in self.to_rows(message):
            for column in row:
                columns[column].append(row[column])
        for column in dtypes:
            columns[column] = pandas.Series(columns[column], dtype=dtypes[column])
        return pandas.DataFrame(columns, columns=self._column_names)

    def _parse_avro_schema(self):
        """Extract and parse Avro schema from a read session."""
        if self._avro_schema_json:
            return

        self._avro_schema_json = json.loads(self._read_session.avro_schema.schema)
        self._column_names = tuple(
            (field["name"] for field in self._avro_schema_json["fields"])
        )

    def _parse_fastavro(self):
        """Convert parsed Avro schema to fastavro format."""
        self._parse_avro_schema()
        self._fastavro_schema = fastavro.parse_schema(self._avro_schema_json)

    def to_rows(self, message):
        """Parse all rows in a stream message.

        Args:
            message ( \
                ~google.cloud.bigquery_storage_v1beta1.types.ReadRowsResponse \
            ):
                A message containing Avro bytes to parse into rows.

        Returns:
            Iterable[Mapping]:
                A sequence of rows, represented as dictionaries.
        """
        self._parse_fastavro()
        messageio = six.BytesIO(message.avro_rows.serialized_binary_rows)
        while True:
            # Loop in a while loop because schemaless_reader can only read
            # a single record.
            try:
                # TODO: Parse DATETIME into datetime.datetime (no timezone),
                #       instead of as a string.
                yield fastavro.schemaless_reader(messageio, self._fastavro_schema)
            except StopIteration:
                break  # Finished with message


def _copy_stream_position(position):
    """Copy a StreamPosition.

    Args:
        position (Union[ \
            dict, \
            ~google.cloud.bigquery_storage_v1beta1.types.StreamPosition \
        ]):
            StreamPostion (or dictionary in StreamPosition format) to copy.

    Returns:
        ~google.cloud.bigquery_storage_v1beta1.types.StreamPosition:
            A copy of the input StreamPostion.
    """
    if isinstance(position, types.StreamPosition):
        output = types.StreamPosition()
        output.CopyFrom(position)
        return output

    return types.StreamPosition(**position)
