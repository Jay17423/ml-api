(ns ml-api.algorithms.ngram-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.ngram :as ng]
   [ml-api.algorithms.tokenizer :as tok]
   [ml-api.utils :as utils]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "NGram executes successfully"
    (with-redefs [tok/transform (fn [_ _ _]
                                  :tokenized-dataset)

                  ng/transform (fn [_ _ _ _]
                                 :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:generated_ngrams
                                          ["hello world"
                                           "world spark"]}])
                  log/info
                  (fn [_])]

      (let [res (ng/execute :dataset
                            {:input_field "text"
                             :output_field "generated_ngrams"
                             :ngram_size 2})]

        (is (= {:data [{:generated_ngrams ["hello world" "world spark"]}]}
               res))))))

(deftest execute-default-values
  (testing "NGram uses default values"
    (with-redefs [ng/transform (fn [_ _ output-field ngram-size]
                                 (is (= "generated_ngrams"
                                        output-field))
                                 (is (= 2
                                        ngram-size))
                                 :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:generated_ngrams
                                          ["sample text"]}])

                  log/info (fn [_])]

      (let [res (ng/execute :dataset
                            {:input_field "text"})]

        (is (= {:data [{:generated_ngrams
                        ["sample text"]}]}
               res))))))

(deftest execute-negative-and-edge
  (testing "Transform failure throws exception"
    (with-redefs [ng/transform (fn [_ _ _ _]
                                 (throw (RuntimeException.
                                         "transform failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"NGram execution failed"

           (ng/execute :dataset
                       {:input_field "text"})))))

  (testing "dataset->json failure throws exception"
    (with-redefs [ng/transform (fn [_ _ _ _]
                                 :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        (throw (RuntimeException.
                                                "json conversion failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"NGram execution failed"
           (ng/execute :dataset
                       {:input_field "text"}))))))
