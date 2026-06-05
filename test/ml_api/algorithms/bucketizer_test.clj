(ns ml-api.algorithms.bucketizer-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.bucketizer :as bucketizer]
   [ml-api.utils :as utils]
   [taoensso.timbre :as log]))

(deftest splits->2d-array-success
  (testing "splits->2d-array converts nested vectors into double arrays"
    (let [result (bucketizer/splits->2d-array [[0.0 1.0 2.0]
                                               [0.0 5.0 10.0]])]
      (is (= 2 (count result))))))

(deftest execute-success-response
  (testing "Bucketizer executes successfully"
    (with-redefs [bucketizer/transform (fn [_ _ _ _ _]
                                         :transformed-dataset)

                  utils/preview-columns (fn [_ _]
                                          [:feature :bucket])

                  utils/dataset->json (fn [_ _]
                                        [{:feature 2.5
                                          :bucket 1.0}])
                  log/info (fn [_])]

      (let [res (bucketizer/execute :dataset
                                    {:feature_field ["feature"]
                                     :output_field ["bucket"]
                                     :bucket_splits [[0.0 1.0 5.0]]
                                     :invalid_value "keep"})]

        (is (= {:data [{:feature 2.5 :bucket 1.0}]} res))))))

(deftest execute-default-invalid-value
  (testing "Bucketizer uses default invalid value"
    (with-redefs [bucketizer/transform (fn [_ _ _ _ invalid-value]
                                         (is (= "error" invalid-value))
                                         :transformed-dataset)

                  utils/preview-columns (fn [_ _]
                                          [:feature :bucket])

                  utils/dataset->json (fn [_ _]
                                        [{:feature 0.5
                                          :bucket 0.0}])
                  log/info (fn [_])]

      (let [res (bucketizer/execute :dataset
                                    {:feature_field ["feature"]
                                     :output_field ["bucket"]
                                     :bucket_splits [[0.0 1.0 5.0]]})]

        (is (= {:data [{:feature 0.5 :bucket 0.0}]} res))))))

(deftest execute-negative-and-edge
  (testing "Transform failure throws exception"

    (with-redefs [bucketizer/transform (fn [_ _ _ _ _]
                                         (throw (RuntimeException.
                                                 "transform failed")))
                  log/info (fn [_])]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Bucketizer execution failed"
           (bucketizer/execute :dataset
                               {:feature_field ["feature"]
                                :output_field ["bucket"]
                                :bucket_splits [[0.0 1.0 5.0]]})))))
  (testing "dataset->json failure throws exception"
    (with-redefs [bucketizer/transform (fn [_ _ _ _ _]
                                         :transformed-dataset)

                  utils/preview-columns (fn [_ _]
                                          [:feature :bucket])

                  utils/dataset->json (fn [_ _]
                                        (throw (RuntimeException.
                                                "json conversion failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Bucketizer execution failed"
           (bucketizer/execute :dataset
                               {:feature_field ["feature"]
                                :output_field ["bucket"]
                                :bucket_splits [[0.0 1.0 5.0]]}))))))
