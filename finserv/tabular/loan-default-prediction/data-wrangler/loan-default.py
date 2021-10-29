import base64
import collections
import io
import os
import re
import logging
import numpy as np
import pandas as pd
import tempfile
import zipfile
from collections import Counter
from contextlib import redirect_stdout
from datetime import date
from enum import Enum
from io import BytesIO
from pyspark.sql import functions as sf, types, Column, Window
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import udf, pandas_udf, to_timestamp
from pyspark.sql.session import SparkSession
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    FractionalType,
    IntegralType,
    LongType,
    StringType,
    TimestampType,
)
from pyspark.sql.utils import AnalysisException
from statsmodels.tsa.seasonal import STL


#  You may want to configure the Spark Context with the right credentials provider.
spark = SparkSession.builder.master("local").getOrCreate()
mode = None

JOIN_COLUMN_LIMIT = 10
ESCAPE_CHAR_PATTERN = re.compile("[{}]+".format(re.escape(".`")))
VALID_JOIN_TYPE = frozenset(
    [
        "anti",
        "cross",
        "full",
        "full_outer",
        "fullouter",
        "inner",
        "left",
        "left_anti",
        "left_outer",
        "left_semi",
        "leftanti",
        "leftouter",
        "leftsemi",
        "outer",
        "right",
        "right_outer",
        "rightouter",
        "semi",
    ],
)


def capture_stdout(func, *args, **kwargs):
    """Capture standard output to a string buffer"""
    stdout_string = io.StringIO()
    with redirect_stdout(stdout_string):
        func(*args, **kwargs)
    return stdout_string.getvalue()


def convert_or_coerce(pandas_df, spark):
    """Convert pandas df to pyspark df and coerces the mixed cols to string"""
    try:
        return spark.createDataFrame(pandas_df)
    except TypeError as e:
        match = re.search(r".*field (\w+).*Can not merge type.*", str(e))
        if match is None:
            raise e
        mixed_col_name = match.group(1)
        # Coercing the col to string
        pandas_df[mixed_col_name] = pandas_df[mixed_col_name].astype("str")
        return pandas_df


def dedupe_columns(cols):
    # if there are duplicate column names after join, append "_0", "_1" to dedupe and mark as renamed.
    # If the original df already takes the name, we will append more "_dup" at the end til it's unique.
    col_to_count = Counter(cols)
    duplicate_col_to_count = {col: col_to_count[col] for col in col_to_count if col_to_count[col] != 1}
    for i in range(len(cols)):
        col = cols[i]
        if col in duplicate_col_to_count:
            idx = col_to_count[col] - duplicate_col_to_count[col]
            new_col_name = f"{col}_{str(idx)}"
            while new_col_name in col_to_count:
                new_col_name += "_dup"
            cols[i] = new_col_name
            duplicate_col_to_count[col] -= 1
    return cols


def default_spark(value):
    return {"default": value}


def default_spark_with_stdout(df, stdout):
    return {
        "default": df,
        "stdout": stdout,
    }


def default_spark_with_trained_parameters(value, trained_parameters):
    return {"default": value, "trained_parameters": trained_parameters}


def default_spark_with_trained_parameters_and_state(df, trained_parameters, state):
    return {"default": df, "trained_parameters": trained_parameters, "state": state}


def dispatch(key_name, args, kwargs, funcs):
    """
    Dispatches to another operator based on a key in the passed parameters.
    This also slices out any parameters using the parameter_name passed in,
    and will reassemble the trained_parameters correctly after invocation.

    Args:
        key_name: name of the key in kwargs used to identify the function to use.
        args: dataframe that will be passed as the first set of parameters to the function.
        kwargs: keyword arguments that key_name will be found in; also where args will be passed to parameters.
                These are also expected to include trained_parameters if there are any.
        funcs: dictionary mapping from value of key_name to (function, parameter_name)

    """
    if key_name not in kwargs:
        raise OperatorCustomerError(f"Missing required parameter {key_name}")

    operator = kwargs[key_name]

    if operator not in funcs:
        raise OperatorCustomerError(f"Invalid choice selected for {key_name}. {operator} is not supported.")

    func, parameter_name = funcs[operator]

    # Extract out the parameters that should be available.
    func_params = kwargs.get(parameter_name, {})
    if func_params is None:
        func_params = {}

    # Extract out any trained parameters.
    specific_trained_parameters = None
    if "trained_parameters" in kwargs:
        trained_parameters = kwargs["trained_parameters"]
        if trained_parameters is not None and parameter_name in trained_parameters:
            specific_trained_parameters = trained_parameters[parameter_name]
    func_params["trained_parameters"] = specific_trained_parameters

    result = spark_operator_with_escaped_column(func, args, func_params)

    # Check if the result contains any trained parameters and remap them to the proper structure.
    if result is not None and "trained_parameters" in result:
        existing_trained_parameters = kwargs.get("trained_parameters")
        updated_trained_parameters = result["trained_parameters"]

        if existing_trained_parameters is not None or updated_trained_parameters is not None:
            existing_trained_parameters = existing_trained_parameters if existing_trained_parameters is not None else {}
            existing_trained_parameters[parameter_name] = result["trained_parameters"]

            # Update the result trained_parameters so they are part of the original structure.
            result["trained_parameters"] = existing_trained_parameters
        else:
            # If the given trained parameters were None and the returned trained parameters were None, don't return
            # anything.
            del result["trained_parameters"]

    return result


def get_and_validate_join_keys(join_keys):
    join_keys_left = []
    join_keys_right = []
    for join_key in join_keys:
        left_key = join_key.get("left", "")
        right_key = join_key.get("right", "")
        if not left_key or not right_key:
            raise OperatorCustomerError("Missing join key: left('{}'), right('{}')".format(left_key, right_key))
        join_keys_left.append(left_key)
        join_keys_right.append(right_key)

    if len(join_keys_left) > JOIN_COLUMN_LIMIT:
        raise OperatorCustomerError("We only support join on maximum 10 columns for one operation.")
    return join_keys_left, join_keys_right


def get_dataframe_with_sequence_ids(df: DataFrame):
    df_cols = df.columns
    rdd_with_seq = df.rdd.zipWithIndex()
    df_with_seq = rdd_with_seq.toDF()
    df_with_seq = df_with_seq.withColumnRenamed("_2", "_seq_id_")
    for col_name in df_cols:
        df_with_seq = df_with_seq.withColumn(col_name, df_with_seq["_1"].getItem(col_name))
    df_with_seq = df_with_seq.drop("_1")
    return df_with_seq


def get_execution_state(status: str, message=None):
    return {"status": status, "message": message}


def rename_invalid_column(df, orig_col):
    """Rename a given column in a data frame to a new valid name

    Args:
        df: Spark dataframe
        orig_col: input column name

    Returns:
        a tuple of new dataframe with renamed column and new column name
    """
    temp_col = orig_col
    if ESCAPE_CHAR_PATTERN.search(orig_col):
        idx = 0
        temp_col = ESCAPE_CHAR_PATTERN.sub("_", orig_col)
        name_set = set(df.columns)
        while temp_col in name_set:
            temp_col = f"{temp_col}_{idx}"
            idx += 1
        df = df.withColumnRenamed(orig_col, temp_col)
    return df, temp_col


def spark_operator_with_escaped_column(operator_func, func_args, func_params):
    """Invoke operator func with input dataframe that has its column names sanitized.

    This function rename column names with special char to an internal name and
    rename it back after invocation

    Args:
        operator_func: underlying operator function
        func_args: operator function positional args, this only contains one element `df` for now
        func_params: operator function kwargs

    Returns:
        a dictionary with operator results
    """
    renamed_columns = {}
    input_keys = ["input_column"]

    for input_col_key in input_keys:
        if input_col_key not in func_params:
            continue
        input_col_value = func_params[input_col_key]
        # rename this col if needed
        input_df, temp_col_name = rename_invalid_column(func_args[0], input_col_value)
        func_args[0] = input_df
        if temp_col_name != input_col_value:
            renamed_columns[input_col_value] = temp_col_name
            func_params[input_col_key] = temp_col_name

    # invoke underlying function
    result = operator_func(*func_args, **func_params)

    # put renamed columns back if applicable
    if result is not None and "default" in result:
        result_df = result["default"]
        # rename col
        for orig_col_name, temp_col_name in renamed_columns.items():
            if temp_col_name in result_df.columns:
                result_df = result_df.withColumnRenamed(temp_col_name, orig_col_name)

        result["default"] = result_df

    return result


def stl_decomposition(ts, period=None):
    """Completes a Season-Trend Decomposition using LOESS (Cleveland et. al. 1990) on time series data.
    
    Parameters
    ----------
    ts: pandas.Series, index must be datetime64[ns] and values must be int or float.
    period: int, primary periodicity of the series. Default is None, will apply a default behavior
        Default behavior:
            if timestamp frequency is minute: period = 1440 / # of minutes between consecutive timestamps
            if timestamp frequency is second: period = 3600 / # of seconds between consecutive timestamps
            if timestamp frequency is ms, us, or ns: period = 1000 / # of ms/us/ns between consecutive timestamps
            else: defer to statsmodels' behavior, detailed here: 
                https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/tsatools.py#L776

    Returns
    -------
    season: pandas.Series, index is same as ts, values are seasonality of ts
    trend: pandas.Series, index is same as ts, values are trend of ts
    resid: pandas.Series, index is same as ts, values are the remainder (original signal, subtract season and trend)
    """
    # TODO: replace this with another, more complex method for finding a better period
    period_sub_hour = {
        "T": 1440,  # minutes
        "S": 3600,  # seconds
        "M": 1000,  # milliseconds
        "U": 1000,  # microseconds
        "N": 1000,  # nanoseconds
    }
    if period is None:
        freq = ts.index.freq
        if freq is None:
            freq = pd.tseries.frequencies.to_offset(pd.infer_freq(ts.index))
        if freq is None:  # if still none, datetimes are not uniform, so raise error
            raise OperatorCustomerError(
                f"{freq} No uniform datetime frequency detected. Make sure the column contains datetimes that are evenly spaced (Are there any missing values?)"
            )
        for k, v in period_sub_hour.items():
            # if freq is not in period_sub_hour, then it is hourly or above and we don't have to set a default
            if k in freq.name:
                period = int(v / int(freq.n))  # n is always >= 1
                break
    decomposition = STL(ts, period=period).fit()
    return decomposition.seasonal, decomposition.trend, decomposition.resid


def to_timestamp_single(x):
    """Helper function for auto-detecting datetime format and casting to ISO-8601 string."""
    converted = pd.to_datetime(x, errors="coerce")
    return converted.astype("str").replace("NaT", "")  # makes pandas NaT into empty string


def uniform_sample(df, target_example_num, n_rows=None, min_required_rows=None):
    if n_rows is None:
        n_rows = df.count()
    if min_required_rows and n_rows < min_required_rows:
        raise OperatorCustomerError(
            f"Not enough valid rows available. Expected a minimum of {min_required_rows}, but the dataset contains "
            f"only {n_rows}"
        )
    sample_ratio = min(1, 3.0 * target_example_num / n_rows)
    return df.sample(withReplacement=False, fraction=float(sample_ratio), seed=0).limit(target_example_num)


def validate_col_name_in_df(col, df_cols):
    if col not in df_cols:
        raise OperatorCustomerError("Cannot resolve column name '{}'.".format(col))


def validate_join_type(join_type):
    if join_type not in VALID_JOIN_TYPE:
        raise OperatorCustomerError(
            "Unsupported join type '{}'. Supported join types include: {}.".format(
                join_type, ", ".join(VALID_JOIN_TYPE)
            )
        )


class OperatorCustomerError(Exception):
    """Error type for Customer Errors in Spark Operators"""


)


def encode_categorical_ordinal_encode(
    df, input_column=None, output_column=None, invalid_handling_strategy=None, trained_parameters=None
):
    INVALID_HANDLING_STRATEGY_SKIP = "Skip"
    INVALID_HANDLING_STRATEGY_ERROR = "Error"
    INVALID_HANDLING_STRATEGY_KEEP = "Keep"
    INVALID_HANDLING_STRATEGY_REPLACE_WITH_NAN = "Replace with NaN"

    from pyspark.ml.feature import StringIndexer, StringIndexerModel
    from pyspark.sql.functions import when

    expects_column(df, input_column, "Input column")

    invalid_handling_map = {
        INVALID_HANDLING_STRATEGY_SKIP: "skip",
        INVALID_HANDLING_STRATEGY_ERROR: "error",
        INVALID_HANDLING_STRATEGY_KEEP: "keep",
        INVALID_HANDLING_STRATEGY_REPLACE_WITH_NAN: "keep",
    }

    output_column, output_is_temp = get_temp_col_if_not_set(df, output_column)

    # process inputs
    handle_invalid = (
        invalid_handling_strategy
        if invalid_handling_strategy in invalid_handling_map
        else INVALID_HANDLING_STRATEGY_ERROR
    )

    trained_parameters = load_trained_parameters(
        trained_parameters, {"invalid_handling_strategy": invalid_handling_strategy}
    )

    input_handle_invalid = invalid_handling_map.get(handle_invalid)
    index_model, index_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, StringIndexerModel, "string_indexer_model"
    )

    if index_model is None:
        indexer = StringIndexer(inputCol=input_column, outputCol=output_column, handleInvalid=input_handle_invalid)
        # fit the model and transform
        try:
            index_model = fit_and_save_model(trained_parameters, "string_indexer_model", indexer, df)
        except Exception as e:
            if input_handle_invalid == "error":
                raise OperatorSparkOperatorCustomerError(
                    f"Encountered error calculating string indexes. Halting because error handling is set to 'Error'. Please check your data and try again: {e}"
                )
            else:
                raise e

    output_df = transform_using_trained_model(index_model, df, index_model_loaded)

    # finally, if missing should be nan, convert them
    if handle_invalid == INVALID_HANDLING_STRATEGY_REPLACE_WITH_NAN:
        new_val = float("nan")
        # convert all numLabels indices to new_val
        num_labels = len(index_model.labels)
        output_df = output_df.withColumn(
            output_column, when(output_df[output_column] == num_labels, new_val).otherwise(output_df[output_column])
        )

    # finally handle the output column name appropriately.
    output_df = replace_input_if_output_is_temp(output_df, input_column, output_column, output_is_temp)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def encode_categorical_one_hot_encode(
    df,
    input_column=None,
    input_already_ordinal_encoded=None,
    invalid_handling_strategy=None,
    drop_last=None,
    output_style=None,
    output_column=None,
    trained_parameters=None,
):

    INVALID_HANDLING_STRATEGY_SKIP = "Skip"
    INVALID_HANDLING_STRATEGY_ERROR = "Error"
    INVALID_HANDLING_STRATEGY_KEEP = "Keep"

    OUTPUT_STYLE_VECTOR = "Vector"
    OUTPUT_STYLE_COLUMNS = "Columns"

    invalid_handling_map = {
        INVALID_HANDLING_STRATEGY_SKIP: "skip",
        INVALID_HANDLING_STRATEGY_ERROR: "error",
        INVALID_HANDLING_STRATEGY_KEEP: "keep",
    }

    handle_invalid = invalid_handling_map.get(invalid_handling_strategy, "error")
    expects_column(df, input_column, "Input column")
    output_format = output_style if output_style in [OUTPUT_STYLE_VECTOR, OUTPUT_STYLE_COLUMNS] else OUTPUT_STYLE_VECTOR
    drop_last = parse_parameter(bool, drop_last, "Drop Last", True)
    input_ordinal_encoded = parse_parameter(bool, input_already_ordinal_encoded, "Input already ordinal encoded", False)

    output_column = output_column if output_column else input_column

    trained_parameters = load_trained_parameters(
        trained_parameters, {"invalid_handling_strategy": invalid_handling_strategy, "drop_last": drop_last}
    )

    from pyspark.ml.feature import (
        StringIndexer,
        StringIndexerModel,
        OneHotEncoder,
        OneHotEncoderModel,
    )
    from pyspark.ml.functions import vector_to_array
    import pyspark.sql.functions as sf
    from pyspark.sql.types import DoubleType

    # first step, ordinal encoding. Not required if input_ordinal_encoded==True
    # get temp name for ordinal encoding
    ordinal_name = temp_col_name(df, output_column)
    if input_ordinal_encoded:
        df_ordinal = df.withColumn(ordinal_name, df[input_column].cast("int"))
        labels = None
    else:
        index_model, index_model_loaded = load_pyspark_model_from_trained_parameters(
            trained_parameters, StringIndexerModel, "string_indexer_model"
        )
        if index_model is None:
            # one hot encoding in PySpark will not work with empty string, replace it with null values
            df = df.withColumn(input_column, sf.when(sf.col(input_column) == "", None).otherwise(sf.col(input_column)))
            # apply ordinal encoding
            indexer = StringIndexer(inputCol=input_column, outputCol=ordinal_name, handleInvalid=handle_invalid)
            try:
                index_model = fit_and_save_model(trained_parameters, "string_indexer_model", indexer, df)
            except Exception as e:
                if handle_invalid == "error":
                    raise OperatorSparkOperatorCustomerError(
                        f"Encountered error calculating string indexes. Halting because error handling is set to 'Error'. Please check your data and try again: {e}"
                    )
                else:
                    raise e

        try:
            df_ordinal = transform_using_trained_model(index_model, df, index_model_loaded)
        except Exception as e:
            if handle_invalid == "error":
                raise OperatorSparkOperatorCustomerError(
                    f"Encountered error transforming string indexes. Halting because error handling is set to 'Error'. Please check your data and try again: {e}"
                )
            else:
                raise e

        labels = index_model.labels

    # drop the input column if required from the ordinal encoded dataset
    if output_column == input_column:
        df_ordinal = df_ordinal.drop(input_column)

    temp_output_col = temp_col_name(df_ordinal, output_column)

    # apply onehot encoding on the ordinal
    cur_handle_invalid = handle_invalid if input_ordinal_encoded else "error"
    cur_handle_invalid = "keep" if cur_handle_invalid == "skip" else cur_handle_invalid

    ohe_model, ohe_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, OneHotEncoderModel, "one_hot_encoder_model"
    )
    if ohe_model is None:
        ohe = OneHotEncoder(
            dropLast=drop_last, handleInvalid=cur_handle_invalid, inputCol=ordinal_name, outputCol=temp_output_col
        )
        try:
            ohe_model = fit_and_save_model(trained_parameters, "one_hot_encoder_model", ohe, df_ordinal)
        except Exception as e:
            if handle_invalid == "error":
                raise OperatorSparkOperatorCustomerError(
                    f"Encountered error calculating encoding categories. Halting because error handling is set to 'Error'. Please check your data and try again: {e}"
                )
            else:
                raise e

    output_df = transform_using_trained_model(ohe_model, df_ordinal, ohe_model_loaded)

    if output_format == OUTPUT_STYLE_COLUMNS:
        if labels is None:
            labels = list(range(ohe_model.categorySizes[0]))

        current_output_cols = set(list(output_df.columns))
        old_cols = [sf.col(escape_column_name(name)) for name in df.columns if name in current_output_cols]
        arr_col = vector_to_array(output_df[temp_output_col])
        new_cols = [(arr_col[i]).alias(f"{output_column}_{name}") for i, name in enumerate(labels)]
        output_df = output_df.select(*(old_cols + new_cols))
    else:
        # remove the temporary ordinal encoding
        output_df = output_df.drop(ordinal_name)
        output_df = output_df.withColumn(output_column, sf.col(temp_output_col))
        output_df = output_df.drop(temp_output_col)
        final_ordering = [col for col in df.columns]
        if output_column not in final_ordering:
            final_ordering.append(output_column)

        final_ordering = escape_column_names(final_ordering)
        output_df = output_df.select(final_ordering)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


import re


def featurize_text_character_statistics_word_count(text: str) -> float:
    return float(len(text.split()))


def featurize_text_character_statistics_char_count(text: str) -> float:
    return float(len(text))


def featurize_text_character_statistics_special_ratio(text: str) -> float:
    text = re.sub("\\s+", "", text)
    if not text:
        return 0.0
    new_str = re.sub(r"[\w]+", "", text)
    return len(new_str) / len(text)


def featurize_text_character_statistics_digit_ratio(text: str) -> float:
    text = re.sub("\\s+", "", text)
    if not text:
        return 0.0
    return sum(c.isdigit() for c in text) / len(text)


def featurize_text_character_statistics_lower_ratio(text: str) -> float:
    text = re.sub("\\s+", "", text)
    if not text:
        return 0.0
    return sum(c.islower() for c in text) / len(text)


def featurize_text_character_statistics_capital_ratio(text: str) -> float:
    text = re.sub("\\s+", "", text)
    if not text:
        return 0.0
    return sum(c.isupper() for c in text) / len(text)


def featurize_text_character_statistics(df, input_column=None, output_column_prefix=None, trained_parameters=None):
    """
    Extracts syntactic features from a string
    Args:
        df: input pyspark dataframe
        spark: spark session
        column: Input column. This column should contain text data
        output_prefix: The output consists of multiple columns, matching the different extracted features.
            This determines their prefix
    Returns:
        dataframe with columns containing features of the string in the input column
            word_count: number of words
            char_count: string length
            special_ratio: ratio of non alphanumeric characters to non-spaces in the string, 0 if empty string
            digit_ratio: ratio of digits characters to non-spaces in the string, 0 if empty string
            lower_ratio: ratio of lowercase characters to non-spaces in the string, 0 if empty string
            capital_ratio: ratio of uppercase characters to non-spaces in the string, 0 if empty string
    """
    from pyspark.sql.functions import udf
    from pyspark.sql.types import DoubleType

    expects_column(df, input_column, "Input column")
    output_column_prefix = output_column_prefix if output_column_prefix else input_column + "_"

    functions = {
        "word_count": featurize_text_character_statistics_word_count,
        "char_count": featurize_text_character_statistics_char_count,
        "special_ratio": featurize_text_character_statistics_special_ratio,
        "digit_ratio": featurize_text_character_statistics_digit_ratio,
        "lower_ratio": featurize_text_character_statistics_lower_ratio,
        "capital_ratio": featurize_text_character_statistics_capital_ratio,
    }

    # cast input to string
    temp_col = temp_col_name(df, *[output_column_prefix + func_name for func_name in functions.keys()])
    output_df = df.withColumn(temp_col, df[input_column].cast("string"))

    # add the features, one at a time
    for func_name, func in functions.items():
        ufunc = udf(func, returnType=DoubleType())
        output_df = output_df.withColumn(output_column_prefix + "_" + func_name, ufunc(output_df[temp_col]))

    output_df = output_df.drop(temp_col)

    return default_spark(output_df)


def featurize_text_vectorize(
    df,
    input_column=None,
    tokenizer=None,
    tokenizer_standard_parameters=None,
    tokenizer_custom_parameters=None,
    vectorizer=None,
    vectorizer_count_vectorizer_parameters=None,
    vectorizer_hashing_parameters=None,
    apply_idf=None,
    apply_idf_yes_parameters=None,
    apply_idf_no_parameters=None,
    output_format=None,
    output_column=None,
    trained_parameters=None,
):

    TOKENIZER_STANDARD = "Standard"
    TOKENIZER_CUSTOM = "Custom"

    VECTORIZER_COUNT_VECTORIZER = "Count Vectorizer"
    VECTORIZER_HASHING = "Hashing"

    APPLY_IDF_YES = "Yes"
    APPLY_IDF_NO = "No"

    OUTPUT_FORMAT_VECTOR = "Vector"
    OUTPUT_FORMAT_COLUMNS = "Columns"

    import re
    from pyspark.ml.feature import (
        Tokenizer,
        RegexTokenizer,
        HashingTF,
        CountVectorizer,
        CountVectorizerModel,
        IDF,
        IDFModel,
    )
    from pyspark.sql.functions import udf, lit
    from pyspark.sql.types import DoubleType, StringType

    expects_column(df, input_column, "Input column")
    expects_parameter_value_in_list("Tokenizer", tokenizer, [TOKENIZER_STANDARD, TOKENIZER_CUSTOM])
    expects_parameter_value_in_list("Vectorizer", vectorizer, [VECTORIZER_COUNT_VECTORIZER, VECTORIZER_HASHING])
    expects_parameter_value_in_list("Apply IDF", apply_idf, [APPLY_IDF_YES, APPLY_IDF_NO])
    expects_parameter_value_in_list("Output format", output_format, [OUTPUT_FORMAT_VECTOR, OUTPUT_FORMAT_COLUMNS])
    output_column = output_column if output_column else f"{input_column}_features"

    column_type = df.schema[input_column].dataType
    if not isinstance(column_type, StringType):
        raise OperatorSparkOperatorCustomerError(
            f'String column required. Please cast column to a string type first. Column "{input_column}" has type {column_type.simpleString()}.'
        )

    if tokenizer == TOKENIZER_CUSTOM:
        expects_parameter("Custom tokenizer parameters", tokenizer_custom_parameters)

    if vectorizer == VECTORIZER_COUNT_VECTORIZER:
        expects_parameter("Count vectorizer parameters", vectorizer_count_vectorizer_parameters)
    elif vectorizer == VECTORIZER_HASHING:
        expects_parameter("Hashing vectorizer parameters", vectorizer_hashing_parameters)

    if apply_idf == APPLY_IDF_YES:
        expects_parameter("Apply IDF parameters", apply_idf_yes_parameters)

    # raise error if vectorizer is not count_vectorizer
    if output_format == OUTPUT_FORMAT_COLUMNS and vectorizer != VECTORIZER_COUNT_VECTORIZER:
        raise OperatorSparkOperatorCustomerError(
            "To produce column output the vectorizer must be a Count Vectorizer. "
            f"Provided vectorizer is {vectorizer}"
        )

    trained_parameters = load_trained_parameters(
        trained_parameters,
        [
            tokenizer,
            tokenizer_standard_parameters,
            tokenizer_custom_parameters,
            vectorizer,
            vectorizer_count_vectorizer_parameters,
            vectorizer_hashing_parameters,
            apply_idf,
            apply_idf_yes_parameters,
            apply_idf_no_parameters,
        ],
    )

    # fill missing value with empty string
    df = df.fillna("", subset=[input_column])

    # tokenize
    token_col = temp_col_name(df, output_column)
    if tokenizer == TOKENIZER_STANDARD:
        tokenize = Tokenizer(inputCol=input_column, outputCol=token_col)
    else:  # custom
        min_token_length = parse_parameter(
            int, tokenizer_custom_parameters.get("minimum_token_length", 1), "minimum_token_length", 1
        )
        gaps = parse_parameter(bool, tokenizer_custom_parameters.get("gaps", True), "gaps", True)
        pattern = str(tokenizer_custom_parameters["pattern"])
        try:
            re.compile(pattern)
        except re.error:
            raise OperatorSparkOperatorCustomerError(
                f"Invalid regex pattern provided. Expected a legal regular expression but input received is {pattern}"
            )
        to_lower_case = parse_parameter(
            bool,
            tokenizer_custom_parameters.get("lowercase_before_tokenization", True),
            "lowercase_before_tokenization",
            True,
        )

        tokenize = RegexTokenizer(
            inputCol=input_column,
            outputCol=token_col,
            minTokenLength=min_token_length,
            gaps=gaps,
            pattern=pattern,
            toLowercase=to_lower_case,
        )
    with_token = tokenize.transform(df)

    # vectorize
    vector_col = temp_col_name(with_token, output_column)
    if vectorizer == VECTORIZER_HASHING:
        # check for legal inputs
        hashing_num_features = parse_parameter(
            int, vectorizer_hashing_parameters.get("number_of_features", 262144), "number_of_features", 262144
        )
        vectorize = HashingTF(inputCol=token_col, outputCol=vector_col, numFeatures=hashing_num_features)
    else:  # "Count vectorizer"
        min_term_freq = parse_parameter(
            float,
            vectorizer_count_vectorizer_parameters.get("minimum_term_frequency", 1.0),
            "minimum_term_frequency",
            1.0,
        )
        min_doc_freq = parse_parameter(
            float,
            vectorizer_count_vectorizer_parameters.get("minimum_document_frequency", 1.0),
            "minimum_document_frequency",
            1.0,
        )
        max_doc_freq = parse_parameter(
            float,
            vectorizer_count_vectorizer_parameters.get("maximum_document_frequency", 1.0),
            "maximum_document_frequency",
            1.0,
        )
        max_vocab_size = parse_parameter(
            int,
            vectorizer_count_vectorizer_parameters.get("maximum_vocabulary_size", 10000),
            "maximum_vocabulary_size",
            10000,
        )
        binary = parse_parameter(
            bool, vectorizer_count_vectorizer_parameters.get("binarize_count", False), "binarize_count", False
        )

        vectorize, vectorize_model_loaded = load_pyspark_model_from_trained_parameters(
            trained_parameters, CountVectorizerModel, "vectorizer_model"
        )

        if vectorize is None:
            count_vectorizer = CountVectorizer(
                inputCol=token_col,
                outputCol=vector_col,
                minTF=min_term_freq,
                minDF=min_doc_freq,
                maxDF=max_doc_freq,
                vocabSize=max_vocab_size,
                binary=binary,
            )
            vectorize = fit_and_save_model(trained_parameters, "vectorizer_model", count_vectorizer, with_token)

    with_vector = vectorize.transform(with_token).drop(token_col)

    # tf-idf
    if apply_idf == APPLY_IDF_YES:
        # check variables
        min_doc_freq = parse_parameter(
            int, apply_idf_yes_parameters.get("minimum_document_frequency", 1), "minimum_document_frequency", 1
        )

        idf_model, idf_model_loaded = load_pyspark_model_from_trained_parameters(
            trained_parameters, IDFModel, "idf_model"
        )

        if idf_model is None:
            idf = IDF(minDocFreq=min_doc_freq, inputCol=vector_col, outputCol=output_column,)
            idf_model = fit_and_save_model(trained_parameters, "idf_model", idf, with_vector)

        post_idf = idf_model.transform(with_vector)
    else:
        post_idf = with_vector.withColumn(output_column, with_vector[vector_col])

    # flatten output if requested
    if output_format == OUTPUT_FORMAT_COLUMNS:
        index_to_name = vectorize.vocabulary

        def indexing(vec, idx):
            try:
                return float(vec[int(idx)])
            except (IndexError, ValueError):
                return 0.0

        indexing_udf = udf(indexing, returnType=DoubleType())

        names = list(df.columns)
        for col_index, cur_name in enumerate(index_to_name):
            names.append(indexing_udf(output_column, lit(col_index)).alias(f"{output_column}_{cur_name}"))

        output_df = post_idf.select(escape_column_names(names))
    else:
        output_df = post_idf.drop(vector_col)

    return default_spark_with_trained_parameters(output_df, trained_parameters)




def manage_columns_drop_column(df, column_to_drop=None, trained_parameters=None):
    expects_column(df, column_to_drop, "Column to drop")
    output_df = df.drop(column_to_drop)
    return default_spark(output_df)


def manage_columns_duplicate_column(df, input_column=None, new_name=None, trained_parameters=None):
    expects_column(df, input_column, "Input column")
    expects_valid_column_name(new_name, "New name")
    if input_column == new_name:
        raise OperatorSparkOperatorCustomerError(
            f"Name for the duplicated column ({new_name}) cannot be the same as the existing column name ({input_column})."
        )

    df = df.withColumn(new_name, df[input_column])
    return default_spark(df)


def manage_columns_rename_column(df, input_column=None, new_name=None, trained_parameters=None):
    expects_column(df, input_column, "Input column")
    expects_valid_column_name(new_name, "New name")

    if input_column == new_name:
        raise OperatorSparkOperatorCustomerError(f"The new name ({new_name}) is the same as the old name ({input_column}).")
    if not new_name:
        raise OperatorSparkOperatorCustomerError(f"Invalid name specified for column {new_name}")

    df = df.withColumnRenamed(input_column, new_name)
    return default_spark(df)


def manage_columns_move_to_start(df, column_to_move=None, trained_parameters=None):
    if column_to_move not in df.columns:
        raise OperatorSparkOperatorCustomerError(f"Invalid column selected to move. Does not exist: {column_to_move}")

    reordered_columns = [df[column_to_move]] + [col for col in df.columns if col != column_to_move]
    df = df.select(escape_column_names(reordered_columns))

    return default_spark(df)


def manage_columns_move_to_end(df, column_to_move=None, trained_parameters=None):
    if column_to_move not in df.columns:
        raise OperatorSparkOperatorCustomerError(f"Invalid column selected to move. Does not exist: {column_to_move}")

    reordered_columns = [col for col in df.columns if col != column_to_move] + [df[column_to_move]]
    df = df.select(escape_column_names(reordered_columns))

    return default_spark(df)


def manage_columns_move_to_index(df, column_to_move=None, index=None, trained_parameters=None):
    index = parse_parameter(int, index, "Index")

    if column_to_move not in df.columns:
        raise OperatorSparkOperatorCustomerError(f"Invalid column selected to move. Does not exist: {column_to_move}")
    if index >= len(df.columns) or index < 0:
        raise OperatorSparkOperatorCustomerError(
            "Specified index must be less than or equal to the number of columns and greater than zero."
        )

    columns_without_move_column = [col for col in df.columns if col != column_to_move]
    reordered_columns = columns_without_move_column[:index] + [column_to_move] + columns_without_move_column[index:]

    df = df.select(escape_column_names(reordered_columns))

    return default_spark(df)


def manage_columns_move_after(df, column_to_move=None, target_column=None, trained_parameters=None):
    if column_to_move not in df.columns:
        raise OperatorSparkOperatorCustomerError(f"Invalid column selected to move. Does not exist: {column_to_move}")

    if target_column not in df.columns:
        raise OperatorSparkOperatorCustomerError(f"Invalid target column selected to move after. Does not exist: {target_column}")

    if column_to_move == target_column:
        raise OperatorSparkOperatorCustomerError(
            f"Invalid reference column name. "
            f"The reference column ({target_column}) should not be the same as the column {column_to_move}."
            f"Use a valid reference column name."
        )

    columns_without_move_column = [col for col in df.columns if col != column_to_move]
    target_index = columns_without_move_column.index(target_column)
    reordered_columns = (
        columns_without_move_column[: (target_index + 1)]
        + [column_to_move]
        + columns_without_move_column[(target_index + 1) :]
    )

    df = df.select(escape_column_names(reordered_columns))
    return default_spark(df)


def manage_columns_move_before(df, column_to_move=None, target_column=None, trained_parameters=None):
    if column_to_move not in df.columns:
        raise OperatorSparkOperatorCustomerError(f"Invalid column selected to move. Does not exist: {column_to_move}")

    if target_column not in df.columns:
        raise OperatorSparkOperatorCustomerError(f"Invalid target column selected to move before. Does not exist: {target_column}")

    if column_to_move == target_column:
        raise OperatorSparkOperatorCustomerError(
            f"Invalid reference column name. "
            f"The reference column ({target_column}) should not be the same as the column {column_to_move}."
            f"Use a valid reference column name."
        )

    columns_without_move_column = [col for col in df.columns if col != column_to_move]
    target_index = columns_without_move_column.index(target_column)
    reordered_columns = (
        columns_without_move_column[:target_index] + [column_to_move] + columns_without_move_column[target_index:]
    )

    df = df.select(escape_column_names(reordered_columns))

    return default_spark(df)


def manage_columns_move_column(df, **kwargs):
    return dispatch(
        "move_type",
        [df],
        kwargs,
        {
            "Move to start": (manage_columns_move_to_start, "move_to_start_parameters"),
            "Move to end": (manage_columns_move_to_end, "move_to_end_parameters"),
            "Move to index": (manage_columns_move_to_index, "move_to_index_parameters"),
            "Move after": (manage_columns_move_after, "move_after_parameters"),
            "Move before": (manage_columns_move_before, "move_before_parameters"),
        },
    )


from pyspark.ml.feature import (
    VectorAssembler,
    StandardScaler,
    StandardScalerModel,
    RobustScaler,
    RobustScalerModel,
    MinMaxScaler,
    MinMaxScalerModel,
    MaxAbsScaler,
    MaxAbsScalerModel,
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as sf
from pyspark.sql.types import NumericType


def process_numeric_standard_scaler(
    df, input_column=None, center=None, scale=None, output_column=None, trained_parameters=None
):
    expects_column(df, input_column, "Input column")
    expects_valid_column_name(output_column, "Output column", nullable=True)
    process_numeric_expects_numeric_column(df, input_column)

    temp_vector_col = temp_col_name(df)
    assembled = VectorAssembler(inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="keep").transform(df)
    assembled_wo_nans = VectorAssembler(
        inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="skip"
    ).transform(df)
    temp_normalized_vector_col = temp_col_name(assembled)

    trained_parameters = load_trained_parameters(
        trained_parameters, {"input_column": input_column, "center": center, "scale": scale}
    )

    scaler_model, scaler_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, StandardScalerModel, "scaler_model"
    )

    if scaler_model is None:
        scaler = StandardScaler(
            inputCol=temp_vector_col,
            outputCol=temp_normalized_vector_col,
            withStd=parse_parameter(bool, scale, "scale", True),
            withMean=parse_parameter(bool, center, "center", False),
        )
        scaler_model = fit_and_save_model(trained_parameters, "scaler_model", scaler, assembled_wo_nans)

    output_df = transform_using_trained_model(scaler_model, assembled, scaler_model_loaded)

    # convert the resulting vector back to numeric
    temp_flattened_vector_col = temp_col_name(output_df)
    output_df = output_df.withColumn(temp_flattened_vector_col, vector_to_array(temp_normalized_vector_col))

    # keep only the final scaled column.
    output_column = input_column if output_column is None or not output_column else output_column
    output_column_value = sf.col(temp_flattened_vector_col)[0].alias(output_column)
    output_df = output_df.withColumn(output_column, output_column_value)
    final_columns = list(dict.fromkeys((list(df.columns) + [output_column])))
    final_columns = escape_column_names(final_columns)
    output_df = output_df.select(final_columns)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def process_numeric_robust_scaler(
    df,
    input_column=None,
    lower_quantile=None,
    upper_quantile=None,
    center=None,
    scale=None,
    output_column=None,
    trained_parameters=None,
):
    expects_column(df, input_column, "Input column")
    expects_valid_column_name(output_column, "Output column", nullable=True)
    process_numeric_expects_numeric_column(df, input_column)

    temp_vector_col = temp_col_name(df)
    assembled = VectorAssembler(inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="keep").transform(df)
    assembled_wo_nans = VectorAssembler(
        inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="skip"
    ).transform(df)
    temp_normalized_vector_col = temp_col_name(assembled)

    trained_parameters = load_trained_parameters(
        trained_parameters,
        {
            "input_column": input_column,
            "center": center,
            "scale": scale,
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
        },
    )

    scaler_model, scaler_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, RobustScalerModel, "scaler_model"
    )

    if scaler_model is None:
        scaler = RobustScaler(
            inputCol=temp_vector_col,
            outputCol=temp_normalized_vector_col,
            lower=parse_parameter(float, lower_quantile, "lower_quantile", 0.25),
            upper=parse_parameter(float, upper_quantile, "upper_quantile", 0.75),
            withCentering=parse_parameter(bool, center, "with_centering", False),
            withScaling=parse_parameter(bool, scale, "with_scaling", True),
        )
        scaler_model = fit_and_save_model(trained_parameters, "scaler_model", scaler, assembled_wo_nans)

    output_df = transform_using_trained_model(scaler_model, assembled, scaler_model_loaded)

    # convert the resulting vector back to numeric
    temp_flattened_vector_col = temp_col_name(output_df)
    output_df = output_df.withColumn(temp_flattened_vector_col, vector_to_array(temp_normalized_vector_col))

    # keep only the final scaled column.
    output_column = input_column if output_column is None or not output_column else output_column
    output_column_value = sf.col(temp_flattened_vector_col)[0].alias(output_column)
    output_df = output_df.withColumn(output_column, output_column_value)
    final_columns = list(dict.fromkeys((list(df.columns) + [output_column])))
    final_columns = escape_column_names(final_columns)
    output_df = output_df.select(final_columns)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def process_numeric_min_max_scaler(
    df, input_column=None, min=None, max=None, output_column=None, trained_parameters=None
):
    expects_column(df, input_column, "Input column")
    expects_valid_column_name(output_column, "Output column", nullable=True)
    process_numeric_expects_numeric_column(df, input_column)

    temp_vector_col = temp_col_name(df)
    assembled = VectorAssembler(inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="keep").transform(df)
    assembled_wo_nans = VectorAssembler(
        inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="skip"
    ).transform(df)
    temp_normalized_vector_col = temp_col_name(assembled)

    trained_parameters = load_trained_parameters(
        trained_parameters, {"input_column": input_column, "min": min, "max": max,}
    )

    scaler_model, scaler_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, MinMaxScalerModel, "scaler_model"
    )

    if scaler_model is None:
        scaler = MinMaxScaler(
            inputCol=temp_vector_col,
            outputCol=temp_normalized_vector_col,
            min=parse_parameter(float, min, "min", 0.0),
            max=parse_parameter(float, max, "max", 1.0),
        )
        scaler_model = fit_and_save_model(trained_parameters, "scaler_model", scaler, assembled_wo_nans)

    output_df = transform_using_trained_model(scaler_model, assembled, scaler_model_loaded)

    # convert the resulting vector back to numeric
    temp_flattened_vector_col = temp_col_name(output_df)
    output_df = output_df.withColumn(temp_flattened_vector_col, vector_to_array(temp_normalized_vector_col))

    # keep only the final scaled column.
    output_column = input_column if output_column is None or not output_column else output_column
    output_column_value = sf.col(temp_flattened_vector_col)[0].alias(output_column)
    output_df = output_df.withColumn(output_column, output_column_value)
    final_columns = list(dict.fromkeys((list(df.columns) + [output_column])))
    final_columns = escape_column_names(final_columns)
    output_df = output_df.select(final_columns)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def process_numeric_max_absolute_scaler(df, input_column=None, output_column=None, trained_parameters=None):
    expects_column(df, input_column, "Input column")
    expects_valid_column_name(output_column, "Output column", nullable=True)
    process_numeric_expects_numeric_column(df, input_column)

    temp_vector_col = temp_col_name(df)
    assembled = VectorAssembler(inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="keep").transform(df)
    assembled_wo_nans = VectorAssembler(
        inputCols=[input_column], outputCol=temp_vector_col, handleInvalid="skip"
    ).transform(df)
    temp_normalized_vector_col = temp_col_name(assembled)

    trained_parameters = load_trained_parameters(trained_parameters, {"input_column": input_column,})

    scaler_model, scaler_model_loaded = load_pyspark_model_from_trained_parameters(
        trained_parameters, MinMaxScalerModel, "scaler_model"
    )

    if scaler_model is None:
        scaler = MinMaxScaler(inputCol=temp_vector_col, outputCol=temp_normalized_vector_col)
        scaler_model = fit_and_save_model(trained_parameters, "scaler_model", scaler, assembled_wo_nans)

    output_df = transform_using_trained_model(scaler_model, assembled, scaler_model_loaded)

    scaler = MaxAbsScaler(inputCol=temp_vector_col, outputCol=temp_normalized_vector_col)

    output_df = scaler.fit(assembled_wo_nans).transform(assembled)

    # convert the resulting vector back to numeric
    temp_flattened_vector_col = temp_col_name(output_df)
    output_df = output_df.withColumn(temp_flattened_vector_col, vector_to_array(temp_normalized_vector_col))

    # keep only the final scaled column.
    output_column = input_column if output_column is None or not output_column else output_column
    output_column_value = sf.col(temp_flattened_vector_col)[0].alias(output_column)
    output_df = output_df.withColumn(output_column, output_column_value)
    final_columns = list(dict.fromkeys((list(df.columns) + [output_column])))
    final_columns = escape_column_names(final_columns)
    output_df = output_df.select(final_columns)

    return default_spark_with_trained_parameters(output_df, trained_parameters)


def process_numeric_expects_numeric_column(df, input_column):
    column_type = df.schema[input_column].dataType
    if not isinstance(column_type, NumericType):
        raise OperatorSparkOperatorCustomerError(
            f'Numeric column required. Please cast column to a numeric type first. Column "{input_column}" has type {column_type.simpleString()}.'
        )


def process_numeric_scale_values(df, **kwargs):
    return dispatch(
        "scaler",
        [df],
        kwargs,
        {
            "Standard scaler": (process_numeric_standard_scaler, "standard_scaler_parameters"),
            "Robust scaler": (process_numeric_robust_scaler, "robust_scaler_parameters"),
            "Min-max scaler": (process_numeric_min_max_scaler, "min_max_scaler_parameters"),
            "Max absolute scaler": (process_numeric_max_absolute_scaler, "max_absolute_scaler_parameters"),
        },
    )




class NonCastableDataHandlingMethod(Enum):
    REPLACE_WITH_NULL = "replace_null"
    REPLACE_WITH_NULL_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN = "replace_null_with_new_col"
    REPLACE_WITH_FIXED_VALUE = "replace_value"
    REPLACE_WITH_FIXED_VALUE_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN = "replace_value_with_new_col"
    DROP_NON_CASTABLE_ROW = "drop"

    @staticmethod
    def get_names():
        return [item.name for item in NonCastableDataHandlingMethod]

    @staticmethod
    def get_values():
        return [item.value for item in NonCastableDataHandlingMethod]


class MohaveDataType(Enum):
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"
    FLOAT = "float"
    LONG = "long"
    STRING = "string"
    OBJECT = "object"

    @staticmethod
    def get_names():
        return [item.name for item in MohaveDataType]

    @staticmethod
    def get_values():
        return [item.value for item in MohaveDataType]


PYTHON_TYPE_MAPPING = {
    MohaveDataType.BOOL: bool,
    MohaveDataType.DATE: str,
    MohaveDataType.DATETIME: str,
    MohaveDataType.FLOAT: float,
    MohaveDataType.LONG: int,
    MohaveDataType.STRING: str,
}

MOHAVE_TO_SPARK_TYPE_MAPPING = {
    MohaveDataType.BOOL: BooleanType,
    MohaveDataType.DATE: DateType,
    MohaveDataType.DATETIME: TimestampType,
    MohaveDataType.FLOAT: DoubleType,
    MohaveDataType.LONG: LongType,
    MohaveDataType.STRING: StringType,
}

SPARK_TYPE_MAPPING_TO_SQL_TYPE = {
    BooleanType: "BOOLEAN",
    LongType: "BIGINT",
    DoubleType: "DOUBLE",
    StringType: "STRING",
    DateType: "DATE",
    TimestampType: "TIMESTAMP",
}

SPARK_TO_MOHAVE_TYPE_MAPPING = {value: key for (key, value) in MOHAVE_TO_SPARK_TYPE_MAPPING.items()}


def cast_column_helper(df, column, mohave_data_type, date_col, datetime_col, non_date_col):
    """Helper for casting a single column to a data type."""
    if mohave_data_type == MohaveDataType.DATE:
        return df.withColumn(column, date_col)
    elif mohave_data_type == MohaveDataType.DATETIME:
        return df.withColumn(column, datetime_col)
    else:
        return df.withColumn(column, non_date_col)


def cast_single_column_type(
    df,
    column,
    mohave_data_type,
    invalid_data_handling_method,
    replace_value=None,
    date_formatting="dd-MM-yyyy",
    datetime_formatting=None,
):
    """Cast single column to a new type

    Args:
        df (DataFrame): spark dataframe
        column (Column): target column for type casting
        mohave_data_type (Enum): Enum MohaveDataType
        invalid_data_handling_method (Enum): Enum NonCastableDataHandlingMethod
        replace_value (str): value to replace for invalid data when "replace_value" is specified
        date_formatting (str): format for date. Default format is "dd-MM-yyyy"
        datetime_formatting (str): format for datetime. Default is None, indicates auto-detection

    Returns:
        df (DataFrame): casted spark dataframe
    """
    cast_to_date = sf.to_date(df[column], date_formatting)
    to_ts = sf.pandas_udf(f=to_timestamp_single, returnType="string")
    if datetime_formatting is None:
        cast_to_datetime = sf.to_timestamp(to_ts(df[column]))  # auto-detect formatting
    else:
        cast_to_datetime = sf.to_timestamp(df[column], datetime_formatting)
    cast_to_non_date = df[column].cast(MOHAVE_TO_SPARK_TYPE_MAPPING[mohave_data_type]())
    non_castable_column = f"{column}_typecast_error"
    temp_column = "temp_column"

    if invalid_data_handling_method == NonCastableDataHandlingMethod.REPLACE_WITH_NULL:
        # Replace non-castable data to None in the same column. pyspark's default behaviour
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | None |
        # | 2 | None |
        # | 3 | 1    |
        # +---+------+
        return cast_column_helper(
            df,
            column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
    if invalid_data_handling_method == NonCastableDataHandlingMethod.DROP_NON_CASTABLE_ROW:
        # Drop non-castable row
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, _ non-castable row
        # +---+----+
        # | id|txt |
        # +---+----+
        # |  3|  1 |
        # +---+----+
        df = cast_column_helper(
            df,
            column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
        return df.where(df[column].isNotNull())

    if (
        invalid_data_handling_method
        == NonCastableDataHandlingMethod.REPLACE_WITH_NULL_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN
    ):
        # Replace non-castable data to None in the same column and put non-castable data to a new column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long
        # +---+----+------------------+
        # | id|txt |txt_typecast_error|
        # +---+----+------------------+
        # |  1|None|      foo         |
        # |  2|None|      bar         |
        # |  3|  1 |                  |
        # +---+----+------------------+
        df = cast_column_helper(
            df,
            temp_column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
        df = df.withColumn(non_castable_column, sf.when(df[temp_column].isNotNull(), "").otherwise(df[column]),)
    elif invalid_data_handling_method == NonCastableDataHandlingMethod.REPLACE_WITH_FIXED_VALUE:
        # Replace non-castable data to a value in the same column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+------+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, replace non-castable value to 0
        # +---+-----+
        # | id| txt |
        # +---+-----+
        # |  1|  0  |
        # |  2|  0  |
        # |  3|  1  |
        # +---+----+
        value = _validate_and_cast_value(value=replace_value, mohave_data_type=mohave_data_type)

        df = cast_column_helper(
            df,
            temp_column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )

        replace_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(
            sf.to_date(sf.lit(value), date_formatting)
        )
        replace_non_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(value)

        df = df.withColumn(
            temp_column, replace_date_value if (mohave_data_type == MohaveDataType.DATE) else replace_non_date_value
        )
    elif (
        invalid_data_handling_method
        == NonCastableDataHandlingMethod.REPLACE_WITH_FIXED_VALUE_AND_PUT_NON_CASTABLE_DATA_IN_NEW_COLUMN
    ):
        # Replace non-castable data to a value in the same column and put non-castable data to a new column
        # Original dataframe
        # +---+------+
        # | id | txt |
        # +---+---+--+
        # | 1 | foo  |
        # | 2 | bar  |
        # | 3 | 1    |
        # +---+------+
        # cast txt column to long, replace non-castable value to 0
        # +---+----+------------------+
        # | id|txt |txt_typecast_error|
        # +---+----+------------------+
        # |  1|  0  |   foo           |
        # |  2|  0  |   bar           |
        # |  3|  1  |                 |
        # +---+----+------------------+
        value = _validate_and_cast_value(value=replace_value, mohave_data_type=mohave_data_type)

        df = cast_column_helper(
            df,
            temp_column,
            mohave_data_type,
            date_col=cast_to_date,
            datetime_col=cast_to_datetime,
            non_date_col=cast_to_non_date,
        )
        df = df.withColumn(non_castable_column, sf.when(df[temp_column].isNotNull(), "").otherwise(df[column]),)

        replace_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(
            sf.to_date(sf.lit(value), date_formatting)
        )
        replace_non_date_value = sf.when(df[temp_column].isNotNull(), df[temp_column]).otherwise(value)

        df = df.withColumn(
            temp_column, replace_date_value if (mohave_data_type == MohaveDataType.DATE) else replace_non_date_value
        )
    # drop temporary column
    df = df.withColumn(column, df[temp_column]).drop(temp_column)

    df_cols = df.columns
    if non_castable_column in df_cols:
        # Arrange columns so that non_castable_column col is next to casted column
        df_cols.remove(non_castable_column)
        column_index = df_cols.index(column)
        arranged_cols = df_cols[: column_index + 1] + [non_castable_column] + df_cols[column_index + 1 :]
        df = df.select(*arranged_cols)
    return df


def _validate_and_cast_value(value, mohave_data_type):
    if value is None:
        return value
    try:
        return PYTHON_TYPE_MAPPING[mohave_data_type](value)
    except ValueError as e:
        raise ValueError(
            f"Invalid value to replace non-castable data. "
            f"{mohave_data_type} is not in mohave supported date type: {MohaveDataType.get_values()}. "
            f"Please use a supported type",
            e,
        )




class OperatorSparkOperatorCustomerError(Exception):
    """Error type for Customer Errors in Spark Operators"""


def temp_col_name(df, *illegal_names):
    """Generates a temporary column name that is unused.
    """
    name = "temp_col"
    idx = 0
    name_set = set(list(df.columns) + list(illegal_names))
    while name in name_set:
        name = f"_temp_col_{idx}"
        idx += 1

    return name


def get_temp_col_if_not_set(df, col_name):
    """Extracts the column name from the parameters if it exists, otherwise generates a temporary column name.
    """
    if col_name:
        return col_name, False
    else:
        return temp_col_name(df), True


def replace_input_if_output_is_temp(df, input_column, output_column, output_is_temp):
    """Replaces the input column in the dataframe if the output was not set

    This is used with get_temp_col_if_not_set to enable the behavior where a 
    transformer will replace its input column if an output is not specified.
    """
    if output_is_temp:
        df = df.withColumn(input_column, df[output_column])
        df = df.drop(output_column)
        return df
    else:
        return df


def parse_parameter(typ, value, key, default=None, nullable=False):
    if value is None:
        if default is not None or nullable:
            return default
        else:
            raise OperatorSparkOperatorCustomerError(f"Missing required input: '{key}'")
    else:
        try:
            value = typ(value)
            if isinstance(value, (int, float, complex)) and not isinstance(value, bool):
                if np.isnan(value) or np.isinf(value):
                    raise OperatorSparkOperatorCustomerError(
                        f"Invalid value provided for '{key}'. Expected {typ.__name__} but received: {value}"
                    )
                else:
                    return value
            else:
                return value
        except (ValueError, TypeError):
            raise OperatorSparkOperatorCustomerError(
                f"Invalid value provided for '{key}'. Expected {typ.__name__} but received: {value}"
            )
        except OverflowError:
            raise OperatorSparkOperatorCustomerError(
                f"Overflow Error: Invalid value provided for '{key}'. Given value '{value}' exceeds the range of type "
                f"'{typ.__name__}' for this input. Insert a valid value for type '{typ.__name__}' and try your request "
                f"again."
            )


def expects_valid_column_name(value, key, nullable=False):
    if nullable and value is None:
        return

    if value is None or len(str(value).strip()) == 0:
        raise OperatorSparkOperatorCustomerError(f"Column name cannot be null, empty, or whitespace for parameter '{key}': {value}")


def expects_parameter(value, key, condition=None):
    if value is None:
        raise OperatorSparkOperatorCustomerError(f"Missing required input: '{key}'")
    elif condition is not None and not condition:
        raise OperatorSparkOperatorCustomerError(f"Invalid value provided for '{key}': {value}")


def expects_column(df, value, key):
    if not value or value not in df.columns:
        raise OperatorSparkOperatorCustomerError(f"Expected column in dataframe for '{key}' however received '{value}'")


def expects_parameter_value_in_list(key, value, items):
    if value not in items:
        raise OperatorSparkOperatorCustomerError(f"Illegal parameter value. {key} expected to be in {items}, but given {value}")


def encode_pyspark_model(model):
    with tempfile.TemporaryDirectory() as dirpath:
        dirpath = os.path.join(dirpath, "model")
        # Save the model
        model.save(dirpath)

        # Create the temporary zip-file.
        mem_zip = BytesIO()
        with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            # Zip the directory.
            for root, dirs, files in os.walk(dirpath):
                for file in files:
                    rel_dir = os.path.relpath(root, dirpath)
                    zf.write(os.path.join(root, file), os.path.join(rel_dir, file))

        zipped = mem_zip.getvalue()
        encoded = base64.b85encode(zipped)
        return str(encoded, "utf-8")


def decode_pyspark_model(model_factory, encoded):
    with tempfile.TemporaryDirectory() as dirpath:
        zip_bytes = base64.b85decode(encoded)
        mem_zip = BytesIO(zip_bytes)
        mem_zip.seek(0)

        with zipfile.ZipFile(mem_zip, "r") as zf:
            zf.extractall(dirpath)

        model = model_factory.load(dirpath)
        return model


def hash_parameters(value):
    # pylint: disable=W0702
    try:
        if isinstance(value, collections.Hashable):
            return hash(value)
        if isinstance(value, dict):
            return hash(frozenset([hash((hash_parameters(k), hash_parameters(v))) for k, v in value.items()]))
        if isinstance(value, list):
            return hash(frozenset([hash_parameters(v) for v in value]))
        raise RuntimeError("Object not supported for serialization")
    except:  # noqa: E722
        raise RuntimeError("Object not supported for serialization")


def load_trained_parameters(trained_parameters, operator_parameters):
    trained_parameters = trained_parameters if trained_parameters else {}
    parameters_hash = hash_parameters(operator_parameters)
    stored_hash = trained_parameters.get("_hash")
    if stored_hash != parameters_hash:
        trained_parameters = {"_hash": parameters_hash}
    return trained_parameters


def load_pyspark_model_from_trained_parameters(trained_parameters, model_factory, name):
    if trained_parameters is None or name not in trained_parameters:
        return None, False

    try:
        model = decode_pyspark_model(model_factory, trained_parameters[name])
        return model, True
    except Exception as e:
        logging.error(f"Could not decode PySpark model {name} from trained_parameters: {e}")
        del trained_parameters[name]
        return None, False


def fit_and_save_model(trained_parameters, name, algorithm, df):
    model = algorithm.fit(df)
    trained_parameters[name] = encode_pyspark_model(model)
    return model


def transform_using_trained_model(model, df, loaded):
    try:
        return model.transform(df)
    except Exception as e:
        if loaded:
            raise OperatorSparkOperatorCustomerError(
                f"Encountered error while using stored model. Please delete the operator and try again. {e}"
            )
        else:
            raise e


ESCAPE_CHAR_PATTERN = re.compile("[{}]+".format(re.escape(".`")))


def escape_column_name(col):
    """Escape column name so it works properly for Spark SQL"""

    # Do nothing for Column object, which should be already valid/quoted
    if isinstance(col, Column):
        return col

    column_name = col

    if ESCAPE_CHAR_PATTERN.search(column_name):
        column_name = f"`{column_name}`"

    return column_name


def escape_column_names(columns):
    return [escape_column_name(col) for col in columns]


def sanitize_df(df):
    """Sanitize dataframe with Spark safe column names and return column name mappings

    Args:
        df: input dataframe

    Returns:
        a tuple of
            sanitized_df: sanitized dataframe with all Spark safe columns
            sanitized_col_mapping: mapping from original col name to sanitized column name
            reversed_col_mapping: reverse mapping from sanitized column name to original col name
    """

    sanitized_col_mapping = {}
    sanitized_df = df

    for orig_col in df.columns:
        if ESCAPE_CHAR_PATTERN.search(orig_col):
            # create a temp column and store the column name mapping
            temp_col = f"{orig_col.replace('.', '_')}_{temp_col_name(sanitized_df)}"
            sanitized_col_mapping[orig_col] = temp_col

            sanitized_df = sanitized_df.withColumn(temp_col, sanitized_df[f"`{orig_col}`"])
            sanitized_df = sanitized_df.drop(orig_col)

    # create a reversed mapping from sanitized col names to original col names
    reversed_col_mapping = {sanitized_name: orig_name for orig_name, sanitized_name in sanitized_col_mapping.items()}

    return sanitized_df, sanitized_col_mapping, reversed_col_mapping


def add_filename_column(df):
    """Add a column containing the input file name of each record."""
    filename_col_name_prefix = "_data_source_filename"
    filename_col_name = filename_col_name_prefix
    counter = 1
    while filename_col_name in df.columns:
        filename_col_name = f"{filename_col_name_prefix}_{counter}"
        counter += 1
    return df.withColumn(filename_col_name, sf.input_file_name())




def type_inference(df):  # noqa: C901 # pylint: disable=R0912
    """Core type inference logic

    Args:
        df: spark dataframe

    Returns: dict a schema that maps from column name to mohave datatype

    """
    columns_to_infer = [col for (col, col_type) in df.dtypes if col_type == "string"]

    pandas_df = df.toPandas()
    report = {}
    for (columnName, _) in pandas_df.iteritems():
        if columnName in columns_to_infer:
            column = pandas_df[columnName].values
            report[columnName] = {
                "sum_string": len(column),
                "sum_numeric": sum_is_numeric(column),
                "sum_integer": sum_is_integer(column),
                "sum_boolean": sum_is_boolean(column),
                "sum_date": sum_is_date(column),
                "sum_datetime": sum_is_datetime(column),
                "sum_null_like": sum_is_null_like(column),
                "sum_null": sum_is_null(column),
            }

    # Analyze
    numeric_threshold = 0.8
    integer_threshold = 0.8
    date_threshold = 0.8
    datetime_threshold = 0.8
    bool_threshold = 0.8

    column_types = {}

    for col, insights in report.items():
        # Convert all columns to floats to make thresholds easy to calculate.
        proposed = MohaveDataType.STRING.value

        sum_is_not_null = insights["sum_string"] - (insights["sum_null"] + insights["sum_null_like"])

        if sum_is_not_null == 0:
            # if entire column is null, keep as string type
            proposed = MohaveDataType.STRING.value
        elif (insights["sum_numeric"] / insights["sum_string"]) > numeric_threshold:
            proposed = MohaveDataType.FLOAT.value
            if (insights["sum_integer"] / insights["sum_numeric"]) > integer_threshold:
                proposed = MohaveDataType.LONG.value
        elif (insights["sum_boolean"] / insights["sum_string"]) > bool_threshold:
            proposed = MohaveDataType.BOOL.value
        elif (insights["sum_date"] / sum_is_not_null) > date_threshold:
            # datetime - date is # of rows with time info
            # if even one value w/ time info in a column with mostly dates, choose datetime
            if (insights["sum_datetime"] - insights["sum_date"]) > 0:
                proposed = MohaveDataType.DATETIME.value
            else:
                proposed = MohaveDataType.DATE.value
        elif (insights["sum_datetime"] / sum_is_not_null) > datetime_threshold:
            proposed = MohaveDataType.DATETIME.value
        column_types[col] = proposed

    for f in df.schema.fields:
        if f.name not in columns_to_infer:
            if isinstance(f.dataType, IntegralType):
                column_types[f.name] = MohaveDataType.LONG.value
            elif isinstance(f.dataType, FractionalType):
                column_types[f.name] = MohaveDataType.FLOAT.value
            elif isinstance(f.dataType, StringType):
                column_types[f.name] = MohaveDataType.STRING.value
            elif isinstance(f.dataType, BooleanType):
                column_types[f.name] = MohaveDataType.BOOL.value
            elif isinstance(f.dataType, TimestampType):
                column_types[f.name] = MohaveDataType.DATETIME.value
            else:
                # unsupported types in mohave
                column_types[f.name] = MohaveDataType.OBJECT.value

    return column_types


def _is_numeric_single(x):
    try:
        x_float = float(x)
        return np.isfinite(x_float)
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False


def sum_is_numeric(x):
    """count number of numeric element

    Args:
        x: numpy array

    Returns: int

    """
    castables = np.vectorize(_is_numeric_single)(x)
    return np.count_nonzero(castables)


def _is_integer_single(x):
    try:
        if not _is_numeric_single(x):
            return False
        return float(x) == int(x)
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False


def sum_is_integer(x):
    castables = np.vectorize(_is_integer_single)(x)
    return np.count_nonzero(castables)


def _is_boolean_single(x):
    boolean_list = ["true", "false"]
    try:
        is_boolean = x.lower() in boolean_list
        return is_boolean
    except ValueError:
        return False
    except TypeError:  # if x = None
        return False
    except AttributeError:
        return False


def sum_is_boolean(x):
    castables = np.vectorize(_is_boolean_single)(x)
    return np.count_nonzero(castables)


def sum_is_null_like(x):  # noqa: C901
    def _is_empty_single(x):
        try:
            return bool(len(x) == 0)
        except TypeError:
            return False

    def _is_null_like_single(x):
        try:
            return bool(null_like_regex.match(x))
        except TypeError:
            return False

    def _is_whitespace_like_single(x):
        try:
            return bool(whitespace_regex.match(x))
        except TypeError:
            return False

    null_like_regex = re.compile(r"(?i)(null|none|nil|na|nan)")  # (?i) = case insensitive
    whitespace_regex = re.compile(r"^\s+$")  # only whitespace

    empty_checker = np.vectorize(_is_empty_single)(x)
    num_is_null_like = np.count_nonzero(empty_checker)

    null_like_checker = np.vectorize(_is_null_like_single)(x)
    num_is_null_like += np.count_nonzero(null_like_checker)

    whitespace_checker = np.vectorize(_is_whitespace_like_single)(x)
    num_is_null_like += np.count_nonzero(whitespace_checker)
    return num_is_null_like


def sum_is_null(x):
    return np.count_nonzero(pd.isnull(x))


def _is_date_single(x):
    try:
        return bool(date.fromisoformat(x))  # YYYY-MM-DD
    except ValueError:
        return False
    except TypeError:
        return False


def sum_is_date(x):
    return np.count_nonzero(np.vectorize(_is_date_single)(x))


def sum_is_datetime(x):
    # detects all possible convertible datetimes, including multiple different formats in the same column
    return pd.to_datetime(x, infer_datetime_format=True, cache=True, errors="coerce").notnull().sum()


def cast_df(df, schema):
    """Cast dataframe from given schema

    Args:
        df: spark dataframe
        schema: schema to cast to. It map from df's col_name to mohave datatype

    Returns: casted dataframe

    """
    # col name to spark data type mapping
    col_to_spark_data_type_map = {}

    # get spark dataframe's actual datatype
    fields = df.schema.fields
    for f in fields:
        col_to_spark_data_type_map[f.name] = f.dataType
    cast_expr = []

    to_ts = pandas_udf(f=to_timestamp_single, returnType="string")

    # iterate given schema and cast spark dataframe datatype
    for col_name in schema:
        mohave_data_type_from_schema = MohaveDataType(schema.get(col_name, MohaveDataType.OBJECT.value))
        if mohave_data_type_from_schema == MohaveDataType.DATETIME:
            df = df.withColumn(col_name, to_timestamp(to_ts(df[col_name])))
            expr = f"`{col_name}`"  # keep the column in the SQL query that is run below
        elif mohave_data_type_from_schema != MohaveDataType.OBJECT:
            spark_data_type_from_schema = MOHAVE_TO_SPARK_TYPE_MAPPING[mohave_data_type_from_schema]
            # Only cast column when the data type in schema doesn't match the actual data type
            if not isinstance(col_to_spark_data_type_map[col_name], spark_data_type_from_schema):
                # use spark-sql expression instead of spark.withColumn to improve performance
                expr = f"CAST (`{col_name}` as {SPARK_TYPE_MAPPING_TO_SQL_TYPE[spark_data_type_from_schema]})"
            else:
                # include column that has same dataType as it is
                expr = f"`{col_name}`"
        else:
            # include column that has same mohave object dataType as it is
            expr = f"`{col_name}`"
        cast_expr.append(expr)
    if len(cast_expr) != 0:
        df = df.selectExpr(*cast_expr)
    return df, schema


def validate_schema(df, schema):
    """Validate if every column is covered in the schema

    Args:
        schema ():
    """
    columns_in_df = df.columns
    columns_in_schema = schema.keys()

    if len(columns_in_df) != len(columns_in_schema):
        raise ValueError(
            f"Invalid schema column size. "
            f"Number of columns in schema should be equal as number of columns in dataframe. "
            f"schema columns size: {len(columns_in_schema)}, dataframe column size: {len(columns_in_df)}"
        )

    for col in columns_in_schema:
        if col not in columns_in_df:
            raise ValueError(
                f"Invalid column name in schema. "
                f"Column in schema does not exist in dataframe. "
                f"Non-existed columns: {col}"
            )


def s3_source(spark, mode, dataset_definition):
    """Represents a source that handles sampling, etc."""


    content_type = dataset_definition["s3ExecutionContext"]["s3ContentType"].upper()
    path = dataset_definition["s3ExecutionContext"]["s3Uri"].replace("s3://", "s3a://")
    recursive = "true" if dataset_definition["s3ExecutionContext"].get("s3DirIncludesNested") else "false"
    adds_filename_column = dataset_definition["s3ExecutionContext"].get("s3AddsFilenameColumn", False)

    try:
        if content_type == "CSV":
            has_header = dataset_definition["s3ExecutionContext"]["s3HasHeader"]
            field_delimiter = dataset_definition["s3ExecutionContext"].get("s3FieldDelimiter", ",")
            if not field_delimiter:
                field_delimiter = ","
            df = spark.read.option("recursiveFileLookup", recursive).csv(
                path=path, header=has_header, escape='"', quote='"', sep=field_delimiter, mode="PERMISSIVE"
            )
        elif content_type == "PARQUET":
            df = spark.read.option("recursiveFileLookup", recursive).parquet(path)
        if adds_filename_column:
            df = add_filename_column(df)
        return default_spark(df)
    except Exception as e:
        raise RuntimeError("An error occurred while reading files from S3") from e


def infer_and_cast_type(df, spark, inference_data_sample_size=1000, trained_parameters=None):
    """Infer column types for spark dataframe and cast to inferred data type.

    Args:
        df: spark dataframe
        spark: spark session
        inference_data_sample_size: number of row data used for type inference
        trained_parameters: trained_parameters to determine if we need infer data types

    Returns: a dict of pyspark df with column data type casted and trained parameters

    """

    # if trained_parameters is none or doesn't contain schema key, then type inference is needed
    if trained_parameters is None or not trained_parameters.get("schema", None):
        # limit first 1000 rows to do type inference

        limit_df = df.limit(inference_data_sample_size)
        schema = type_inference(limit_df)
    else:
        schema = trained_parameters["schema"]
        try:
            validate_schema(df, schema)
        except ValueError as e:
            raise OperatorCustomerError(e)
    try:
        df, schema = cast_df(df, schema)
    except (AnalysisException, ValueError) as e:
        raise OperatorCustomerError(e)
    trained_parameters = {"schema": schema}
    return default_spark_with_trained_parameters(df, trained_parameters)


def join_tables(left_df, right_df, spark, left_column, right_column, join_type):
    """Join two datasets based on single column.
    For duplicate columns after the join, we will append *_0, *_1 to dedupe.

    Args:
        left_df: first dataframe
        right_df: second dataframe
        spark: spark context
        left_columns: column key of first dataframe to join on
        right_columns: column key of second dataframe to join on
        join_type: string, must be one of pyspark's join type

    Returns: joined dataframe

    """

    left_table = left_df.alias("left_table")
    right_table = right_df.alias("right_table")

    validate_join_type(join_type)
    validate_col_name_in_df(left_column, left_df.columns)
    validate_col_name_in_df(right_column, right_df.columns)

    output_df = left_table.join(
        right_table, sf.col(f"left_table.`{left_column}`") == sf.col(f"right_table.`{right_column}`"), how=join_type
    )

    # if there are duplicate column names after join, append *_0, *_1 to dedupe.
    cols = output_df.columns
    if len(cols) != len(set(cols)):
        cols = dedupe_columns(cols)

    return default_spark(output_df.toDF(*cols))


def manage_columns(df, spark, **kwargs):

    return dispatch(
        "operator",
        [df],
        kwargs,
        {
            "Drop column": (manage_columns_drop_column, "drop_column_parameters"),
            "Duplicate column": (manage_columns_duplicate_column, "duplicate_column_parameters"),
            "Rename column": (manage_columns_rename_column, "rename_column_parameters"),
            "Move column": (manage_columns_move_column, "move_column_parameters"),
        },
    )


def process_numeric(df, spark, **kwargs):

    return dispatch(
        "operator", [df], kwargs, {"Scale values": (process_numeric_scale_values, "scale_values_parameters"),},
    )


def encode_categorical(df, spark, **kwargs):

    return dispatch(
        "operator",
        [df],
        kwargs,
        {
            "Ordinal encode": (encode_categorical_ordinal_encode, "ordinal_encode_parameters"),
            "One-hot encode": (encode_categorical_one_hot_encode, "one_hot_encode_parameters"),
        },
    )


def featurize_text(df, spark, **kwargs):

    return dispatch(
        "operator",
        [df],
        kwargs,
        {
            "Character statistics": (featurize_text_character_statistics, "character_statistics_parameters"),
            "Vectorize": (featurize_text_vectorize, "vectorize_parameters"),
        },
    )


op_1_output = s3_source(spark=spark, mode=mode, **{'dataset_definition': {'__typename': 'S3CreateDatasetDefinitionOutput', 'datasetSourceType': 'S3', 'name': 'loans-part-1.csv', 'description': None, 's3ExecutionContext': {'__typename': 'S3ExecutionContext', 's3Uri': 's3://sagemaker-us-east-1-892313895307/loan-default/loans-part-1.csv', 's3ContentType': 'csv', 's3HasHeader': True, 's3FieldDelimiter': ','}}})
op_2_output = infer_and_cast_type(op_1_output['default'], spark=spark, **{})
op_3_output = s3_source(spark=spark, mode=mode, **{'dataset_definition': {'__typename': 'S3CreateDatasetDefinitionOutput', 'datasetSourceType': 'S3', 'name': 'loans-part-2.csv', 'description': None, 's3ExecutionContext': {'__typename': 'S3ExecutionContext', 's3Uri': 's3://sagemaker-us-east-1-892313895307/loan-default/loans-part-2.csv', 's3ContentType': 'csv', 's3HasHeader': True, 's3FieldDelimiter': ','}}})
op_4_output = infer_and_cast_type(op_3_output['default'], spark=spark, **{})
op_5_output = join_tables(op_2_output['default'], op_4_output['default'], spark=spark, **{'left_column': 'id', 'right_column': 'id', 'join_type': 'inner'})
op_6_output = manage_columns(op_5_output['default'], spark=spark, **{'operator': 'Drop column', 'drop_column_parameters': {'column_to_drop': 'id_0'}})
op_7_output = manage_columns(op_6_output['default'], spark=spark, **{'operator': 'Drop column', 'drop_column_parameters': {'column_to_drop': 'id_1'}})
op_11_output = process_numeric(op_7_output['default'], spark=spark, **{'operator': 'Scale values', 'scale_values_parameters': {'scaler': 'Min-max scaler', 'min_max_scaler_parameters': {'min': 0, 'max': 1, 'input_column': 'interest_rate'}, 'standard_scaler_parameters': {'input_column': 'interest_rate', 'scale': True}}})
op_12_output = encode_categorical(op_11_output['default'], spark=spark, **{'operator': 'One-hot encode', 'one_hot_encode_parameters': {'invalid_handling_strategy': 'Keep', 'drop_last': False, 'output_style': 'Vector', 'input_column': 'purpose'}, 'ordinal_encode_parameters': {'invalid_handling_strategy': 'Replace with NaN'}})
op_13_output = featurize_text(op_12_output['default'], spark=spark, **{'operator': 'Vectorize', 'vectorize_parameters': {'tokenizer': 'Standard', 'vectorizer': 'Count Vectorizer', 'apply_idf': 'Yes', 'output_format': 'Vector', 'tokenizer_standard_parameters': {}, 'vectorizer_count_vectorizer_parameters': {'minimum_term_frequency': 1, 'minimum_document_frequency': 1, 'maximum_document_frequency': 0.999, 'maximum_vocabulary_size': 262144, 'binarize_count': False}, 'apply_idf_yes_parameters': {'minimum_document_frequency': 5}, 'input_column': 'employer_title'}, 'character_statistics_parameters': {}})

#  Glossary: variable name to node_id
#
#  op_1_output: 64b38ed2-ccde-4717-add6-3f081a867a44
#  op_2_output: 1010dd6b-c210-4c7b-bb2c-0df68020c012
#  op_3_output: 5d6601f8-31b4-449a-9450-413d29b37431
#  op_4_output: 62d7443b-a708-4dca-9925-4617c24869d4
#  op_5_output: 62dbf6ea-eead-46db-b765-f2c232823327
#  op_6_output: 3f78e8dc-2fb1-4cfe-85e5-8ee2ebdc3917
#  op_7_output: d357df9c-b829-4d5d-8dae-eb7c9dc2e04a
#  op_11_output: 28d240cf-bd04-4fb6-a9dd-795448c8550c
#  op_12_output: cecf08fd-164d-4f31-9488-e636000587d1
#  op_13_output: 4dd276b3-990b-4e44-b4cd-1d3e05771396