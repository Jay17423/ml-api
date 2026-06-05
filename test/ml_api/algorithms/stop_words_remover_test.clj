(ns ml-api.algorithms.stop-words-remover-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.stop-words-remover :as swr]
   [ml-api.algorithms.tokenizer :as tok]
   [ml-api.utils :as utils]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "StopWordsRemover executes successfully"
    (with-redefs [tok/transform (fn [_ _ _]
                                  :tokenized-dataset)
                  swr/transform (fn [_ _ _ _ _]
                                  :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:filtered_words
                                          ["spark" "ml"]}])

                  log/info (fn [_])]

      (let [res (swr/execute :dataset
                             {:input_field "text"
                              :output_field "filtered_words"
                              :stop_words ["the" "is"]
                              :case_sensitive false})]

        (is (= {:data [{:filtered_words ["spark" "ml"]}]}
               res))))))

(deftest execute-default-values
  (testing "StopWordsRemover uses default values"
    (with-redefs [swr/transform (fn [_ _ output-field _ case-sensitive]
                                  (is (= "filtered_words"
                                         output-field))
                                  (is (= false
                                         case-sensitive))
                                  :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:filtered_words
                                          ["hello" "world"]}])
                  log/info (fn [_])]

      (let [res (swr/execute :dataset {:input_field "text"})]

        (is (= {:data [{:filtered_words  ["hello" "world"]}]}
               res))))))

(deftest execute-negative-and-edge
  (testing "Transform failure throws exception"
    (with-redefs [swr/transform (fn [_ _ _ _ _]
                                  (throw (RuntimeException.
                                          "transform failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"StopWordsRemover execution failed"

           (swr/execute :dataset
                        {:input_field "text"})))))

  (testing "dataset->json failure throws exception"
    (with-redefs [swr/transform (fn [_ _ _ _ _]
                                  :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        (throw (RuntimeException.
                                                "json conversion failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"StopWordsRemover execution failed"

           (swr/execute :dataset
                        {:input_field "text"}))))))
