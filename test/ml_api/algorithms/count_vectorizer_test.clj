(ns ml-api.algorithms.count-vectorizer-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.count-vectorizer :as cv]
   [ml-api.algorithms.tokenizer :as tokenizer]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "CountVectorizer executes successfully"
    (with-redefs [tokenizer/transform (fn [_ _ _]
                                        :tokenized-dataset)

                  cv/transform (fn [_ _ _ _ _ _]
                                 :transformed-dataset)

                  cv/dataset->json
                  (fn [_ _ _]
                    [{:words ["hello" "world"]
                      :features "(2,[0,1],[1.0,1.0])"}])

                  log/info (fn [_])]

      (let [res (cv/execute :dataset
                            {:input_field "text"
                             :output_field "features"
                             :word_limit 100
                             :min_docs 1
                             :min_count 1.0})]

        (is (= {:data [{:words ["hello" "world"]
                        :features "(2,[0,1],[1.0,1.0])"}]}
               res))))))

(deftest execute-default-values
  (testing "CountVectorizer uses default values"
    (with-redefs [cv/transform
                  (fn [_ _ output-field word-limit min-docs min-count]
                    (is (= "features" output-field))
                    (is (= 262144 word-limit))
                    (is (= 1 min-docs))
                    (is (= 1.0 min-count))
                    :transformed-dataset)

                  cv/dataset->json
                  (fn [_ _ _]
                    [{:words ["sample"]
                      :features "(1,[0],[1.0])"}])

                  log/info (fn [_])]

      (let [res (cv/execute :dataset
                            {:input_field "text"})]

        (is (= {:data [{:words ["sample"] :features "(1,[0],[1.0])"}]} res))))))

(deftest execute-negative-and-edge
  (testing "Transform failure throws exception"
    (with-redefs [cv/transform (fn [_ _ _ _ _ _]
                                 (throw (RuntimeException.
                                         "transform failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"CountVectorizer execution failed"
           (cv/execute :dataset
                       {:input_field "text"})))))

  (testing "dataset->json failure throws exception"
    (with-redefs [cv/transform (fn [_ _ _ _ _ _]
                                 :transformed-dataset)

                  cv/dataset->json
                  (fn [_ _ _]
                    (throw (RuntimeException.
                            "json conversion failed")))

                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"CountVectorizer execution failed"
           (cv/execute :dataset
                       {:input_field "text"}))))))
