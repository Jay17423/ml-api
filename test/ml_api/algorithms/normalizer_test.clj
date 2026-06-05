(ns ml-api.algorithms.normalizer-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.normalizer :as norm]
   [ml-api.algorithms.vector-assembler :as va]
   [ml-api.utils :as utils]
   [taoensso.timbre :as log]))

(deftest parse-norm-value-success
  (testing "parse-norm-value converts numeric value"
    (is (= 2.0 (norm/parse-norm-value 2.0))))

  (testing "parse-norm-value converts inf into infinity"
    (is (= Double/POSITIVE_INFINITY (norm/parse-norm-value "inf")))))

(deftest execute-success-response
  (testing "Normalizer executes successfully"

    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  norm/transform (fn [_ _ _ _]
                                   :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:normalized_features [0.5 0.5]}])

                  log/info (fn [_])]

      (let [res (norm/execute :dataset
                              {:feature_field ["feature1" "feature2"]
                               :output_field "normalized_features"
                               :norm_value 2.0})]

        (is (= {:data [{:normalized_features  [0.5 0.5]}]}
               res))))))

(deftest execute-default-values
  (testing "Normalizer uses default values"
    (with-redefs [norm/transform (fn [_ _ output-field norm-value]
                                   (is (= "normalized_features"
                                          output-field))
                                   (is (= 2.0
                                          norm-value))
                                   :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:normalized_features [1.0 0.0]}])

                  log/info (fn [_])]

      (let [res (norm/execute :dataset
                              {:feature_field ["feature1"]})]

        (is (= {:data [{:normalized_features [1.0 0.0]}]} res))))))

(deftest execute-negative-and-edge
  (testing "Transform failure throws exception"
    (with-redefs [norm/transform (fn [_ _ _ _]
                                   (throw (RuntimeException.
                                           "transform failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Normalizer execution failed"

           (norm/execute :dataset
                         {:feature_field ["feature1"]})))))

  (testing "dataset->json failure throws exception"
    (with-redefs [norm/transform (fn [_ _ _ _]
                                   :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        (throw (RuntimeException.
                                                "json conversion failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Normalizer execution failed"
           (norm/execute :dataset
                         {:feature_field ["feature1"]}))))))
