#  Copyright 2017-2022 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from pyspark.ml import PipelineModel
from sparknlp.base import HasRecursiveTransform

from sparknlp.common import AnnotatorProperties


class RecursivePipelineModel(PipelineModel):
    """Fitted RecursivePipeline.

    Behaves the same as a Spark PipelineModel does. Not intended to be
    initialized by itself. To create a RecursivePipelineModel please fit data to
    a :class:`.RecursivePipeline`.
    """

    def __init__(self, pipeline_model):
        super(PipelineModel, self).__init__()
        self.stages = pipeline_model.stages

    def _transform(self, dataset):
        for t in self.stages:
            if isinstance(t, HasRecursiveTransform):
                # drops current stage from the recursive pipeline within
                dataset = t.transform_recursive(
                    dataset, PipelineModel(self.stages[:-1])
                )
            elif isinstance(t, AnnotatorProperties) and t.getLazyAnnotator():
                pass
            else:
                dataset = t.transform(dataset)
        return dataset
