(ns ml-api.algorithms.string-indexer-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.string-indexer :as si]
   [ml-api.utils :as utils]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "StringIndexer executes successfully"
    (with-redefs [si/transform (fn [_ _ _ _ _]
                                 :transformed-dataset)

                  utils/preview-columns (fn [_ _]
                                          [:category :category_index])

                  utils/dataset->json (fn [_ _]
                                        [{:category "A"
                                          :category_index 0.0}])

                  log/info (fn [_])]

      (let [res (si/execute :dataset
                            {:input_fields ["category"]
                             :output_fields ["category_index"]
                             :invalid_value "keep"
                             :label_order "frequencyDesc"})]

        (is (= {:data [{:category "A" :category_index 0.0}]}
               res))))))

(deftest execute-default-values
  (testing "StringIndexer uses default values"
    (with-redefs [si/transform (fn [_ _ _ invalid-value label-order]
                                 (is (= "error"
                                        invalid-value))
                                 (is (= "frequencyDesc"
                                        label-order))
                                 :transformed-dataset)

                  utils/preview-columns (fn [_ _]
                                          [:category :category_index])

                  utils/dataset->json (fn [_ _]
                                        [{:category "B"
                                          :category_index 1.0}])
                  log/info (fn [_])]

      (let [res (si/execute :dataset
                            {:input_fields ["category"]
                             :output_fields ["category_index"]})]

        (is (= {:data [{:category "B" :category_index 1.0}]}
               res))))))

(deftest execute-negative-and-edge
  (testing "Transform failure throws exception"
    (with-redefs [si/transform (fn [_ _ _ _ _]
                                 (throw (RuntimeException.
                                         "transform failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"StringIndexer execution failed"
           (si/execute :dataset
                       {:input_fields ["category"]
                        :output_fields ["category_index"]})))))

  (testing "dataset->json failure throws exception"
    (with-redefs [si/transform (fn [_ _ _ _ _]
                                 :transformed-dataset)

                  utils/preview-columns (fn [_ _]
                                          [:category :category_index])

                  utils/dataset->json (fn [_ _]
                                        (throw (RuntimeException.
                                                "json conversion failed")))

                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"StringIndexer execution failed"
           (si/execute :dataset
                       {:input_fields ["category"]
                        :output_fields ["category_index"]}))))))
