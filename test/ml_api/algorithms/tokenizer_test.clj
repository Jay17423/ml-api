(ns ml-api.algorithms.tokenizer-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.tokenizer :as tok]
   [ml-api.utils :as utils]
   [taoensso.timbre :as log]))

(deftest execute-tokenizer-success-response
  (testing "Tokenizer executes successfully"
    (with-redefs [tok/transform (fn [_ _ _]
                                  :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:text "hello world"
                                          :tokens ["hello" "world"]}])

                  log/info (fn [_])]

      (let [res (tok/execute-tokenizer :dataset
                                       {:input_field "text"
                                        :output_field "tokens"})]

        (is (= {:data [{:text "hello world" :tokens ["hello" "world"]}]}
               res))))))

(deftest execute-tokenizer-negative-and-edge
  (testing "Tokenizer transform failure throws exception"
    (with-redefs [tok/transform (fn [_ _ _]
                                  (throw (RuntimeException.
                                          "transform failed")))

                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Tokenizer execution failed"
           (tok/execute-tokenizer :dataset
                                  {:input_field "text"
                                   :output_field "tokens"})))))

  (testing "Tokenizer dataset->json failure throws exception"
    (with-redefs [tok/transform (fn [_ _ _]
                                  :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        (throw (RuntimeException.
                                                "json conversion failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Tokenizer execution failed"

           (tok/execute-tokenizer :dataset
                                  {:input_field "text"
                                   :output_field "tokens"}))))))

(deftest execute-regex-tokenizer-success-response
  (testing "RegexTokenizer executes successfully"
    (with-redefs [tok/regex-transform (fn [_ _ _ _ _ _ _]
                                        :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:text "hello world"
                                          :tokens ["hello" "world"]}])
                  log/info (fn [_])]

      (let [res (tok/execute-regex-tokenizer :dataset
                                             {:input_field "text"
                                              :output_field "tokens"
                                              :pattern "\\s+"
                                              :pattern_as_delimiter true
                                              :minimum_token_length 1
                                              :convert_to_lowercase true})]

        (is (= {:data [{:text "hello world" :tokens ["hello" "world"]}]}
               res))))))

(deftest execute-regex-tokenizer-default-values
  (testing "RegexTokenizer uses default values"

    (with-redefs [tok/regex-transform
                  (fn [_ _ _ pattern pattern-as-delimiter
                       minimum-token-length convert-to-lowercase]

                    (is (= "\\s+" pattern))
                    (is (= true pattern-as-delimiter))
                    (is (= 1 minimum-token-length))
                    (is (= true convert-to-lowercase))
                    :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:tokens ["sample"]}])
                  log/info (fn [_])]

      (let [res (tok/execute-regex-tokenizer :dataset
                                             {:input_field "text"
                                              :output_field "tokens"})]

        (is (= {:data [{:tokens ["sample"]}]}
               res))))))

(deftest execute-regex-tokenizer-negative-and-edge
  (testing "RegexTokenizer transform failure throws exception"
    (with-redefs [tok/regex-transform (fn [_ _ _ _ _ _ _]
                                        (throw (RuntimeException.
                                                "regex transform failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"RegexTokenizer execution failed"

           (tok/execute-regex-tokenizer :dataset
                                        {:input_field "text"
                                         :output_field "tokens"})))))

  (testing "RegexTokenizer dataset->json failure throws exception"
    (with-redefs [tok/regex-transform (fn [_ _ _ _ _ _ _]
                                        :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        (throw (RuntimeException.
                                                "json conversion failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"RegexTokenizer execution failed"
           (tok/execute-regex-tokenizer :dataset
                                        {:input_field "text"
                                         :output_field "tokens"}))))))
