# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LexFiles: English Multinational Legal Corpora"""

import datasets
import json
import os

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """LexFiles: English Multinational Legal Corpora"""

_HOMEPAGE = "https://huggingface.co/lexlms"

_VERSION = "1.0.0"


class LexFilesConfig(datasets.BuilderConfig):
    """BuilderConfig for LexFiles Corpora"""
    def __init__(self, url, **kwargs):
        super(LexFilesConfig, self).__init__(**kwargs)
        self.url = url


class LexFiles(datasets.GeneratorBasedBuilder):
    """
    The LexFiles
    """

    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIG_CLASS = LexFilesConfig
    BUILDER_CONFIGS = [
        LexFilesConfig(
            name="eu-legislation",
            description="EU Legislation",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lex_files/resolve/main/eurlex.zip'
        ),
        LexFilesConfig(
            name="eu-court-cases",
            description="EU Court cases",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lex_files/resolve/main/eurlex.zip'
        ),
        LexFilesConfig(
            name="legal-c4",
            description="Legal webpages extracted from C4",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lexfiles4/resolve/main/legal_c4.zip'
        ),
        LexFilesConfig(
            name="ecthr-cases",
            description="ECtHR cases",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lex_files/resolve/main/ecthr_cases.zip'
        ),
        LexFilesConfig(
            name="uk-legislation",
            description="UK Legislation",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lex_files/resolve/main/uk_legislation.zip'
        ),
        LexFilesConfig(
            name="uk-court-cases",
            description="UK Court cases",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lex_files/resolve/main/uk_courts_cases.zip'
        ),
        LexFilesConfig(
            name="indian-court-cases",
            description="Indian Court cases",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lex_files/resolve/main/indian_courts_cases.zip'
        ),
        LexFilesConfig(
            name="us-contracts",
            description="US Contracts",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lex_files/resolve/main/us_contracts.zip'
        ),
        LexFilesConfig(
            name="us-court-cases",
            description="US Court cases",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lex_files/resolve/main/courtlistener.zip'
        ),
        LexFilesConfig(
            name="us-legislation",
            description="US Legislation",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lex_files/resolve/main/us_legislation.zip'
        ),
        LexFilesConfig(
            name="canadian-legislation",
            description="Canadian Legislation",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lex_files/resolve/main/canadian_legislation.zip'
        ),
        LexFilesConfig(
            name="canadian-court-cases",
            description="Canadian Legislation",
            version=datasets.Version(_VERSION, ""),
            url='https://huggingface.co/datasets/lexlms/lex_files/resolve/main/canadian_court_cases.zip'
        ),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(self.config.url)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(data_dir, "train.jsonl"), "split": 'train'},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(data_dir, "validation.jsonl"), "split": 'validation'},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": os.path.join(data_dir, "test.jsonl"), "split": 'test'},
            ),
        ]

    def _generate_examples(self, filepath, split):
        """
        Reads line by line samples and generates examples.
        :param filepath: Path to jsonl files with line by line examples.
        """
        logger.info("⏳ Generating examples from = %s", filepath)
        if self.config.name in ['eu-court-cases', 'eu-legislation']:
            with open(filepath, encoding="utf-8") as f:
                for idx, row in enumerate(f):
                    data = json.loads(row)
                    if (self.config.name == 'eu-court-cases' and data['sector'] == "6") or \
                            (self.config.name == 'eu-legislation' and data['sector'] != "6"):
                        yield data["id"], {
                            "text": data["text"]
                        }
        else:
            with open(filepath, encoding="utf-8") as f:
                for idx, row in enumerate(f):
                    data = json.loads(row)
                    yield f'{self.config.name}-{split}-{idx}', {
                        "text": data["text"]
                    }