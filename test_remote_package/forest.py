import collections
import datetime
import doctest
import functools
import hashlib
import importlib
import inspect
import io
import itertools
import json
import logging
import logging.config
import math
import os
import pickle
import re
import string
import subprocess
import sys
import time
import traceback
import types
import urllib.request as urllib2
from functools import lru_cache, partial, singledispatch
from logging.handlers import SocketHandler
from pprint import pprint

import geopandas as gpd
import numpy as np
import pandas as pd
import polyline
import pyspark
import pyspark.ml
import rich
import rich.console
import rich.syntax
import shapely
import yaml
from munch import Munch
from objprint import op
from polyline import decode as pl_decode
from polyline import encode as pl_encode
from pyspark.context import SparkContext
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.base import Estimator, Model, Transformer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, Evaluator
from pyspark.ml.feature import (CountVectorizer, HashingTF,
                                QuantileDiscretizer, StandardScaler, Tokenizer,
                                VectorAssembler)
from pyspark.ml.linalg import DenseVector, Vectors, VectorUDT
from pyspark.ml.param import Params
from pyspark.ml.param.shared import (HasInputCol, HasInputCols, HasOutputCol,
                                     HasOutputCols, Param, Params,
                                     TypeConverters)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame, GroupedData, Row, SparkSession
from pyspark.sql import functions as fn
from pyspark.sql.functions import expr as sql_
from pyspark.sql.types import *
from termcolor import colored