(ns ml-api.algorithms.standard-scaler-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.standard-scaler :as ss]
   [ml-api.algorithms.vector-assembler :as va]
   [ml-api.utils :as utils]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "StandardScaler executes successfully"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  ss/transform (fn [_ _ _ _ _]
                                 :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:scaled_features
                                          [0.2 0.8]}])

                  log/info (fn [_])]

      (let [res (ss/execute :dataset
                            {:feature_field ["feature1" "feature2"]
                             :output_field "scaled_features"
                             :with_std true
                             :with_mean false})]

        (is (= {:data [{:scaled_features [0.2 0.8]}]} res))))))

(deftest execute-default-values
  (testing "StandardScaler uses default values"
    (with-redefs [ss/transform (fn [_ _ output-field with-std with-mean]
                                 (is (= "scaled_features"
                                        output-field))
                                 (is (= true
                                        with-std))
                                 (is (= false
                                        with-mean))
                                 :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:scaled_features
                                          [1.0 0.0]}])

                  log/info (fn [_])]

      (let [res (ss/execute :dataset
                            {:feature_field ["feature1"]})]

        (is (= {:data [{:scaled_features [1.0 0.0]}]} res))))))

(deftest execute-negative-and-edge
  (testing "Transform failure throws exception"
    (with-redefs [ss/transform (fn [_ _ _ _ _]
                                 (throw (RuntimeException.
                                         "transform failed")))

                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"StandardScaler execution failed"

           (ss/execute :dataset
                       {:feature_field ["feature1"]})))))

  (testing "dataset->json failure throws exception"
    (with-redefs [ss/transform (fn [_ _ _ _ _]
                                 :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        (throw (RuntimeException.
                                                "json conversion failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"StandardScaler execution failed"
           (ss/execute :dataset
                       {:feature_field ["feature1"]}))))))
