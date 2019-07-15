# -*- coding: utf-8 -*-
#
# Copyright 2019 Google LLC
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

"""Unit tests."""

import mock
import pytest

from google.cloud import automl_v1beta1
from google.api_core import exceptions

PROJECT='project'
REGION='region'
LOCATION_PATH='projects/{}/locations/{}'.format(PROJECT, REGION)

class TestTablesClient(object):

    def tables_client(self, client_attrs={},
            prediction_client_attrs={}):
        client_mock = mock.Mock(**client_attrs)
        prediction_client_mock = mock.Mock(**prediction_client_attrs)
        return automl_v1beta1.TablesClient(client=client_mock,
                prediction_client=prediction_client_mock,
                project=PROJECT, region=REGION)

    def test_list_datasets_empty(self):
        client = self.tables_client({
            'list_datasets.return_value': [],
            'location_path.return_value': LOCATION_PATH,
        }, {})
        ds = client.list_datasets()
        client.client.location_path.assert_called_with(PROJECT, REGION)
        client.client.list_datasets.assert_called_with(LOCATION_PATH)
        assert ds == []

    def test_list_datasets_not_empty(self):
        datasets = ['some_dataset']
        client = self.tables_client({
            'list_datasets.return_value': datasets,
            'location_path.return_value': LOCATION_PATH,
        }, {})
        ds = client.list_datasets()
        client.client.location_path.assert_called_with(PROJECT, REGION)
        client.client.list_datasets.assert_called_with(LOCATION_PATH)
        assert len(ds) == 1
        assert ds[0] == 'some_dataset'

    def test_get_dataset_name(self):
        dataset_actual = 'dataset'
        client = self.tables_client({
            'get_dataset.return_value': dataset_actual
            }, {})
        dataset = client.get_dataset(dataset_name='my_dataset')
        client.client.get_dataset.assert_called_with('my_dataset')
        assert dataset == dataset_actual

    def test_get_no_dataset(self):
        client = self.tables_client({
            'get_dataset.side_effect': exceptions.NotFound('err')
        }, {})
        error = None
        try:
            client.get_dataset(dataset_name='my_dataset')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.get_dataset.assert_called_with('my_dataset')

    def test_get_dataset_from_empty_list(self):
        client = self.tables_client({'list_datasets.return_value': []}, {})
        error = None
        try:
            client.get_dataset(dataset_display_name='my_dataset')
        except exceptions.NotFound as e:
            error = e
        assert error is not None

    def test_get_dataset_from_list_not_found(self):
        client = self.tables_client({
            'list_datasets.return_value': [mock.Mock(display_name='not_it')]
        }, {})
        error = None
        try:
            client.get_dataset(dataset_display_name='my_dataset')
        except exceptions.NotFound as e:
            error = e
        assert error is not None

    def test_get_dataset_from_list(self):
        client = self.tables_client({
            'list_datasets.return_value': [
                mock.Mock(display_name='not_it'),
                mock.Mock(display_name='my_dataset'),
            ]
        }, {})
        dataset = client.get_dataset(dataset_display_name='my_dataset')
        assert dataset.display_name == 'my_dataset'

    def test_create_dataset(self):
        client = self.tables_client({
            'location_path.return_value': LOCATION_PATH,
            'create_dataset.return_value': mock.Mock(display_name='name'),
        }, {})
        metadata = {'metadata': 'values'}
        dataset = client.create_dataset('name', metadata=metadata)
        client.client.location_path.assert_called_with(PROJECT, REGION)
        client.client.create_dataset.assert_called_with(
                LOCATION_PATH,
                {'display_name': 'name', 'tables_dataset_metadata': metadata}
        )
        assert dataset.display_name == 'name'

    def test_delete_dataset(self):
        dataset = mock.Mock()
        dataset.configure_mock(name='name')
        client = self.tables_client({
            'delete_dataset.return_value': None,
        }, {})
        client.delete_dataset(dataset=dataset)
        client.client.delete_dataset.assert_called_with('name')

    def test_delete_dataset_not_found(self):
        client = self.tables_client({
            'list_datasets.return_value': [],
        }, {})
        client.delete_dataset(dataset_display_name='not_found')
        client.client.delete_dataset.assert_not_called()

    def test_delete_dataset_name(self):
        client = self.tables_client({
            'delete_dataset.return_value': None,
        }, {})
        client.delete_dataset(dataset_name='name')
        client.client.delete_dataset.assert_called_with('name')

    def test_import_not_found(self):
        client = self.tables_client({
            'list_datasets.return_value': [],
        }, {})
        error = None
        try:
            client.import_data(dataset_display_name='name', gcs_input_uris='uri')
        except exceptions.NotFound as e:
            error = e
        assert error is not None

        client.client.import_data.assert_not_called()

    def test_import_gcs_uri(self):
        client = self.tables_client({
            'import_data.return_value': None,
        }, {})
        client.import_data(dataset_name='name', gcs_input_uris='uri')
        client.client.import_data.assert_called_with('name', {
            'gcs_source': {'input_uris': ['uri']}
        })

    def test_import_gcs_uris(self):
        client = self.tables_client({
            'import_data.return_value': None,
        }, {})
        client.import_data(dataset_name='name',
                gcs_input_uris=['uri', 'uri'])
        client.client.import_data.assert_called_with('name', {
            'gcs_source': {'input_uris': ['uri', 'uri']}
        })

    def test_import_bq_uri(self):
        client = self.tables_client({
            'import_data.return_value': None,
        }, {})
        client.import_data(dataset_name='name',
                bigquery_input_uri='uri')
        client.client.import_data.assert_called_with('name', {
            'bigquery_source': {'input_uri': 'uri'}
        })

    def test_list_table_specs(self):
        client = self.tables_client({
            'list_table_specs.return_value': None,
        }, {})
        client.list_table_specs(dataset_name='name')
        client.client.list_table_specs.assert_called_with('name')

    def test_list_table_specs_not_found(self):
        client = self.tables_client({
            'list_table_specs.side_effect': exceptions.NotFound('not found'),
        }, {})
        error = None
        try:
            client.list_table_specs(dataset_name='name')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.list_table_specs.assert_called_with('name')

    def test_list_column_specs(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [],
        }, {})
        client.list_column_specs(dataset_name='name')
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')

    def test_update_column_spec_not_found(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        data_type_mock = mock.Mock(type_code='type_code')
        column_spec_mock.configure_mock(name='column', display_name='column',
                data_type=data_type_mock)
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        error = None
        try:
            client.update_column_spec(dataset_name='name',
                    column_spec_name='column2')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_column_spec.assert_not_called()

    def test_update_column_spec_display_name_not_found(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        data_type_mock = mock.Mock(type_code='type_code')
        column_spec_mock.configure_mock(name='column', display_name='column',
                data_type=data_type_mock)
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        error = None
        try:
            client.update_column_spec(dataset_name='name',
                    column_spec_display_name='column2')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_column_spec.assert_not_called()

    def test_update_column_spec_name_no_args(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        data_type_mock = mock.Mock(type_code='type_code')
        column_spec_mock.configure_mock(name='column/2', display_name='column',
                data_type=data_type_mock)
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        client.update_column_spec(dataset_name='name',
                column_spec_name='column/2')
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_column_spec.assert_called_with({
            'name': 'column/2',
            'data_type': {
                'type_code': 'type_code',
            }
        })

    def test_update_column_spec_no_args(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        data_type_mock = mock.Mock(type_code='type_code')
        column_spec_mock.configure_mock(name='column', display_name='column',
                data_type=data_type_mock)
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        client.update_column_spec(dataset_name='name',
                column_spec_display_name='column')
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_column_spec.assert_called_with({
            'name': 'column',
            'data_type': {
                'type_code': 'type_code',
            }
        })

    def test_update_column_spec_nullable(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        data_type_mock = mock.Mock(type_code='type_code')
        column_spec_mock.configure_mock(name='column', display_name='column',
                data_type=data_type_mock)
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        client.update_column_spec(dataset_name='name',
                column_spec_display_name='column', nullable=True)
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_column_spec.assert_called_with({
            'name': 'column',
            'data_type': {
                'type_code': 'type_code',
                'nullable': True,
            }
        })

    def test_update_column_spec_type_code(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        data_type_mock = mock.Mock(type_code='type_code')
        column_spec_mock.configure_mock(name='column', display_name='column',
                data_type=data_type_mock)
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        client.update_column_spec(dataset_name='name',
                column_spec_display_name='column', type_code='type_code2')
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_column_spec.assert_called_with({
            'name': 'column',
            'data_type': {
                'type_code': 'type_code2',
            }
        })

    def test_update_column_spec_type_code_nullable(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        data_type_mock = mock.Mock(type_code='type_code')
        column_spec_mock.configure_mock(name='column', display_name='column',
                data_type=data_type_mock)
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        client.update_column_spec(dataset_name='name', nullable=True,
                column_spec_display_name='column', type_code='type_code2')
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_column_spec.assert_called_with({
            'name': 'column',
            'data_type': {
                'type_code': 'type_code2',
                'nullable': True,
            }
        })

    def test_update_column_spec_type_code_nullable_false(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        data_type_mock = mock.Mock(type_code='type_code')
        column_spec_mock.configure_mock(name='column', display_name='column',
                data_type=data_type_mock)
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        client.update_column_spec(dataset_name='name', nullable=False,
                column_spec_display_name='column', type_code='type_code2')
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_column_spec.assert_called_with({
            'name': 'column',
            'data_type': {
                'type_code': 'type_code2',
                'nullable': False,
            }
        })

    def test_set_target_column_table_not_found(self):
        client = self.tables_client({
            'list_table_specs.side_effect': exceptions.NotFound('err'),
        }, {})
        error = None
        try:
            client.set_target_column(dataset_name='name',
                    column_spec_display_name='column2')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_not_called()
        client.client.update_dataset.assert_not_called()

    def test_set_target_column_not_found(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        column_spec_mock.configure_mock(name='column/1', display_name='column')
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        error = None
        try:
            client.set_target_column(dataset_name='name',
                    column_spec_display_name='column2')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_dataset.assert_not_called()

    def test_set_target_column(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        column_spec_mock.configure_mock(name='column/1', display_name='column')
        dataset_mock = mock.Mock()
        tables_dataset_metadata_mock = mock.Mock()
        tables_dataset_metadata_mock.configure_mock(target_column_spec_id='2',
            weight_column_spec_id='2',
            ml_use_column_spec_id='3')
        dataset_mock.configure_mock(name='dataset',
            tables_dataset_metadata=tables_dataset_metadata_mock)
        client = self.tables_client({
            'get_dataset.return_value': dataset_mock,
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        client.set_target_column(dataset_name='name',
                column_spec_display_name='column')
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_dataset.assert_called_with({
            'name': 'dataset',
            'tables_dataset_metadata': {
                'target_column_spec_id': '1',
                'weight_column_spec_id': '2',
                'ml_use_column_spec_id': '3',
            }
        })

    def test_set_weight_column_table_not_found(self):
        client = self.tables_client({
            'list_table_specs.side_effect': exceptions.NotFound('err'),
        }, {})
        try:
            client.set_weight_column(dataset_name='name',
                    column_spec_display_name='column2')
        except exceptions.NotFound:
            pass
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_not_called()
        client.client.update_dataset.assert_not_called()

    def test_set_weight_column_not_found(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        column_spec_mock.configure_mock(name='column/1', display_name='column')
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        error = None
        try:
            client.set_weight_column(dataset_name='name',
                    column_spec_display_name='column2')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_dataset.assert_not_called()

    def test_set_weight_column(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        column_spec_mock.configure_mock(name='column/2', display_name='column')
        dataset_mock = mock.Mock()
        tables_dataset_metadata_mock = mock.Mock()
        tables_dataset_metadata_mock.configure_mock(target_column_spec_id='1',
            weight_column_spec_id='1',
            ml_use_column_spec_id='3')
        dataset_mock.configure_mock(name='dataset',
            tables_dataset_metadata=tables_dataset_metadata_mock)
        client = self.tables_client({
            'get_dataset.return_value': dataset_mock,
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        client.set_weight_column(dataset_name='name',
                column_spec_display_name='column')
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_dataset.assert_called_with({
            'name': 'dataset',
            'tables_dataset_metadata': {
                'target_column_spec_id': '1',
                'weight_column_spec_id': '2',
                'ml_use_column_spec_id': '3',
            }
        })

    def test_set_test_train_column_table_not_found(self):
        client = self.tables_client({
            'list_table_specs.side_effect': exceptions.NotFound('err'),
        }, {})
        error = None
        try:
            client.set_test_train_column(dataset_name='name',
                    column_spec_display_name='column2')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_not_called()
        client.client.update_dataset.assert_not_called()

    def test_set_test_train_column_not_found(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        column_spec_mock.configure_mock(name='column/1', display_name='column')
        client = self.tables_client({
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        error = None
        try:
            client.set_test_train_column(dataset_name='name',
                    column_spec_display_name='column2')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_dataset.assert_not_called()

    def test_set_test_train_column(self):
        table_spec_mock = mock.Mock()
        # name is reserved in use of __init__, needs to be passed here
        table_spec_mock.configure_mock(name='table')
        column_spec_mock = mock.Mock()
        column_spec_mock.configure_mock(name='column/3', display_name='column')
        dataset_mock = mock.Mock()
        tables_dataset_metadata_mock = mock.Mock()
        tables_dataset_metadata_mock.configure_mock(target_column_spec_id='1',
            weight_column_spec_id='2',
            ml_use_column_spec_id='2')
        dataset_mock.configure_mock(name='dataset',
            tables_dataset_metadata=tables_dataset_metadata_mock)
        client = self.tables_client({
            'get_dataset.return_value': dataset_mock,
            'list_table_specs.return_value': [table_spec_mock],
            'list_column_specs.return_value': [column_spec_mock],
        }, {})
        client.set_test_train_column(dataset_name='name',
                column_spec_display_name='column')
        client.client.list_table_specs.assert_called_with('name')
        client.client.list_column_specs.assert_called_with('table')
        client.client.update_dataset.assert_called_with({
            'name': 'dataset',
            'tables_dataset_metadata': {
                'target_column_spec_id': '1',
                'weight_column_spec_id': '2',
                'ml_use_column_spec_id': '3',
            }
        })

    def test_list_models_empty(self):
        client = self.tables_client({
            'list_models.return_value': [],
            'location_path.return_value': LOCATION_PATH,
        }, {})
        ds = client.list_models()
        client.client.location_path.assert_called_with(PROJECT, REGION)
        client.client.list_models.assert_called_with(LOCATION_PATH)
        assert ds == []

    def test_list_models_not_empty(self):
        models = ['some_model']
        client = self.tables_client({
            'list_models.return_value': models,
            'location_path.return_value': LOCATION_PATH,
        }, {})
        ds = client.list_models()
        client.client.location_path.assert_called_with(PROJECT, REGION)
        client.client.list_models.assert_called_with(LOCATION_PATH)
        assert len(ds) == 1
        assert ds[0] == 'some_model'

    def test_get_model_name(self):
        model_actual = 'model'
        client = self.tables_client({
            'get_model.return_value': model_actual
            }, {})
        model = client.get_model(model_name='my_model')
        client.client.get_model.assert_called_with('my_model')
        assert model == model_actual

    def test_get_no_model(self):
        client = self.tables_client({
            'get_model.side_effect': exceptions.NotFound('err')
        }, {})
        error = None
        try:
            client.get_model(model_name='my_model')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.get_model.assert_called_with('my_model')

    def test_get_model_from_empty_list(self):
        client = self.tables_client({'list_models.return_value': []}, {})
        error = None
        try:
            client.get_model(model_display_name='my_model')
        except exceptions.NotFound as e:
            error = e
        assert error is not None

    def test_get_model_from_list_not_found(self):
        client = self.tables_client({
            'list_models.return_value': [mock.Mock(display_name='not_it')]
        }, {})
        error = None
        try:
            client.get_model(model_display_name='my_model')
        except exceptions.NotFound as e:
            error = e
        assert error is not None

    def test_get_model_from_list(self):
        client = self.tables_client({
            'list_models.return_value': [
                mock.Mock(display_name='not_it'),
                mock.Mock(display_name='my_model'),
            ]
        }, {})
        model = client.get_model(model_display_name='my_model')
        assert model.display_name == 'my_model'

    def test_delete_model(self):
        model = mock.Mock()
        model.configure_mock(name='name')
        client = self.tables_client({
            'delete_model.return_value': None,
        }, {})
        client.delete_model(model=model)
        client.client.delete_model.assert_called_with('name')

    def test_delete_model_not_found(self):
        client = self.tables_client({
            'list_models.return_value': [],
        }, {})
        client.delete_model(model_display_name='not_found')
        client.client.delete_model.assert_not_called()

    def test_delete_model_name(self):
        client = self.tables_client({
            'delete_model.return_value': None,
        }, {})
        client.delete_model(model_name='name')
        client.client.delete_model.assert_called_with('name')

    def test_deploy_model(self):
        client = self.tables_client({}, {})
        client.deploy_model(model_name='name')
        client.client.deploy_model.assert_called_with('name')

    def test_deploy_model_not_found(self):
        client = self.tables_client({
            'list_models.return_value': [],
        }, {})
        error = None
        try:
            client.deploy_model(model_display_name='name')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.deploy_model.assert_not_called()
            
    def test_undeploy_model(self):
        client = self.tables_client({}, {})
        client.undeploy_model(model_name='name')
        client.client.undeploy_model.assert_called_with('name')

    def test_undeploy_model_not_found(self):
        client = self.tables_client({
            'list_models.return_value': [],
        }, {})
        error = None
        try:
            client.undeploy_model(model_display_name='name')
        except exceptions.NotFound as e:
            error = e
        assert error is not None
        client.client.undeploy_model.assert_not_called()
