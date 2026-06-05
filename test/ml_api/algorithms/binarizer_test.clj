(ns ml-api.algorithms.binarizer-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.binarizer :as bin]
   [ml-api.utils :as utils]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "Binarizer executes successfully"
    (with-redefs [bin/transform (fn [_ _ _ _]
                                        :transformed-dataset)
                  utils/preview-columns (fn [_ _]
                                          [:feature :binary-feature])
                  utils/dataset->json (fn [_ _]
                                        [{:feature 0.7
                                          :binary-feature 1.0}])
                  log/info (fn [_])]

      (let [res (bin/execute :dataset
                                   {:input_field ["feature"]
                                    :output_field ["binary-feature"]
                                    :threshold_values [0.5]})]
        (is (= {:data [{:feature 0.7 :binary-feature 1.0}]} res))))))

(deftest execute-default-threshold
  (testing "Binarizer uses default threshold value"
    (with-redefs [bin/transform (fn [_ _ _ threshold-values]
                                        (is (= [0.0] threshold-values))
                                        :transformed-dataset)
                  utils/preview-columns (fn [_ _]
                                          [:feature :binary-feature])
                  utils/dataset->json (fn [_ _]
                                        [{:feature 0.2
                                          :binary-feature 1.0}])
                  log/info (fn [_])]

      (let [res (bin/execute :dataset
                                   {:input_field ["feature"]
                                    :output_field ["binary-feature"]})]
        (is (= {:data [{:feature 0.2 :binary-feature 1.0}]} res))))))

(deftest execute-negative-and-edge
  (testing "Transform failure throws exception"
    (with-redefs [bin/transform (fn [_ _ _ _]
                                        (throw (RuntimeException.
                                                "transform failed")))
                  log/info (fn [_])]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Binarizer execution failed"
           (bin/execute :dataset
                              {:input_field ["feature"]
                               :output_field ["binary-feature"]
                               :threshold_values [0.5]})))))
  (testing "dataset->json failure throws exception"
    (with-redefs [bin/transform (fn [_ _ _ _]
                                        :transformed-dataset)

                  utils/preview-columns (fn [_ _]
                                          [:feature :binary-feature])

                  utils/dataset->json (fn [_ _]
                                        (throw (RuntimeException.
                                                "json conversion failed")))
                  log/info
                  (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Binarizer execution failed"
           (bin/execute :dataset
                              {:input_field ["feature"]
                               :output_field ["binary-feature"]
                               :threshold_values [0.5]}))))))
