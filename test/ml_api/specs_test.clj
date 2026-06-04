(ns ml-api.specs-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.specs :as specs]))

(deftest validate-request-success-test
  (testing "Valid Tokenizer request passes validation"
    (is (nil? (specs/validate-request
               {:algorithm "Tokenizer"
                :parameters {:url "/tmp/data.csv"
                             :input_field "text"
                             :output_field "tokens"}}))))

  (testing "Valid GLR request passes validation"
    (is (nil? (specs/validate-request
               {:algorithm "GeneralizedLinearRegression"
                :parameters {:url "/tmp/data.csv"
                             :feature_field ["age" "salary"]
                             :target_field "price"}})))))

(deftest validate-request-negative-test
  (testing "Throws exception when algorithm is not string"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"algorithm must be string"

                          (specs/validate-request
                           {:algorithm 123
                            :parameters {}}))))

  (testing "Throws exception when parameters is not map"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"parameters must be object"

                          (specs/validate-request
                           {:algorithm "Tokenizer"
                            :parameters []}))))

  (testing "Throws exception for unsupported algorithm"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"Unsupported algorithm"

                          (specs/validate-request
                           {:algorithm "UnknownAlgo"
                            :parameters {}}))))

  (testing "Throws exception for missing required parameter"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"Missing required parameter"

                          (specs/validate-request
                           {:algorithm "Tokenizer"
                            :parameters {:url "/tmp/data.csv"
                                         :input_field "text"}}))))

  (testing "Throws exception for invalid parameter datatype"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"Invalid parameter datatype"

                          (specs/validate-request
                           {:algorithm "CountVectorizer"
                            :parameters {:url "/tmp/data.csv"
                                         :input_field "text"
                                         :output_field "features"
                                         :word_limit "100"}}))))

  (testing "Throws exception for invalid enum value"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"Invalid parameter datatype"

                          (specs/validate-request
                           {:algorithm "GeneralizedLinearRegression"
                            :parameters {:url "/tmp/data.csv"
                                         :feature_field ["age"]
                                         :target_field "price"
                                         :regression_family
                                         "invalid-family"}})))))

(deftest validate-required-parameters-test
  (testing "Validates required parameters successfully"
    (is (nil? (specs/validate-required-parameters
               {:url {:validator string?
                      :type "string"}}
               {:url "/tmp/data.csv"}))))

  (testing "Throws exception for missing required parameter"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"Missing required parameter"

                          (specs/validate-required-parameters
                           {:url {:validator string?
                                  :type "string"}}
                           {})))))

(deftest validate-optional-parameters-test

  (testing "Valid optional parameter passes validation"
    (is (nil? (specs/validate-optional-parameters
               {:max_iterations {:validator int?
                                 :type "integer"}}
               {:max_iterations 10}))))

  (testing "Invalid optional parameter throws exception"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"Invalid parameter datatype"

                          (specs/validate-optional-parameters
                           {:max_iterations {:validator int?
                                             :type "integer"}}
                           {:max_iterations "ten"})))))

(deftest validate-parameter-type-test

  (testing "Valid datatype passes validation"

    (is (nil? (specs/validate-parameter-type :age int? "integer" 10))))

  (testing "Invalid datatype throws exception"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo
                          #"Invalid parameter datatype"

                          (specs/validate-parameter-type
                           :age int? "integer" "ten")))))