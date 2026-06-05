(ns ml-api.algorithms.lda-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.lda :as lda]
   [ml-api.algorithms.count-vectorizer :as cv]
   [ml-api.algorithms.stop-words-remover :as swr]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "LDA executes successfully"
    (with-redefs [swr/transform (fn [_ _ _ _ _]
                                  :filtered-dataset)

                  cv/fit-transform (fn [_ _ _ _ _ _]
                                     {:dataset :vectorized-dataset
                                      :model :cv-model})

                  lda/dataset->json (fn [_ _ _]
                                      [{:input_text "hello world"
                                        :topic_distribution [0.1 0.9]}])

                  lda/topics->json (fn [_ _]
                                     [{:topic 0
                                       :words ["hello" "world"]
                                       :term_indices [0 1]
                                       :term_weights [0.5 0.5]}])

                  log/info (fn [_])

                  lda/execute (fn [_ _]
                                {:data [{:input_text "hello world"
                                         :topic_distribution [0.1 0.9]}]
                                 :topics [{:topic 0
                                           :words ["hello" "world"]
                                           :term_indices [0 1]
                                           :term_weights [0.5 0.5]}]})]

      (let [res (lda/execute :dataset {:text_field "text" :topic_count 5})]
        (is (= 1 (count (:data res))))
        (is (= 1 (count (:topics res))))))))

(deftest execute-default-values
  (testing "LDA uses default values"
    (with-redefs [swr/transform (fn [_ _ _ _ _]
                                  :filtered-dataset)

                  cv/fit-transform (fn [_ _ _ word-limit min-docs min-count]
                                     (is (= 262144 word-limit))
                                     (is (= 1 min-docs))
                                     (is (= 1.0 min-count))
                                     {:dataset :vectorized-dataset
                                      :model :cv-model})

                  lda/dataset->json (fn [_ _ _]
                                      [])

                  lda/topics->json (fn [_ _]
                                     [])

                  log/info (fn [_])

                  lda/execute (fn [_ _]
                                {:data []
                                 :topics []})]

      (let [res (lda/execute :dataset {:text_field "text"})]
        (is (= [] (:data res)))
        (is (= [] (:topics res)))))))

(deftest execute-negative-and-edge
  (testing "Stop words remover failure throws exception"
    (with-redefs [swr/transform (fn [_ _ _ _ _]
                                  (throw (RuntimeException.
                                          "stop words failed")))

                  lda/execute (fn [_ _]
                                (throw (ex-info
                                        "LDA execution failed"
                                        {})))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"LDA execution failed"

           (lda/execute :dataset {:text_field "text"})))))

  (testing "CountVectorizer failure throws exception"
    (with-redefs [swr/transform (fn [_ _ _ _ _]
                                  :filtered-dataset)

                  cv/fit-transform (fn [_ _ _ _ _ _]
                                     (throw (RuntimeException.
                                             "count vectorizer failed")))

                  lda/execute (fn [_ _]
                                (throw (ex-info
                                        "LDA execution failed"
                                        {})))

                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"LDA execution failed"
           (lda/execute :dataset {:text_field "text"}))))))
