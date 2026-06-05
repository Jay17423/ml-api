(ns ml-api.algorithms.glr-model-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.glr-model :as glr]
   [ml-api.utils :as utils]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "GeneralizedLinearRegressionModel executes successfully"
    (with-redefs [glr/transform (fn [_ _ _]
                                  :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        [{:label 10.0 :prediction 9.8}])

                  log/info (fn [_])]

      (let [res (glr/execute :dataset
                             {:model_path "/tmp/model"
                              :feature_field ["feature1" "feature2"]
                              :target_field "label"
                              :output_field "prediction"})]

        (is (= {:data [{:label 10.0 :prediction 9.8}]} res))))))

(deftest execute-default-values
  (testing "GeneralizedLinearRegressionModel uses default values"
    (with-redefs [glr/transform (fn [_ model-path feature-field]
                                  (is (= "/tmp/model" model-path))
                                  (is (= ["feature1"] feature-field))
                                  :transformed-dataset)

                  utils/dataset->json (fn [_ columns]
                                        (is (= ["label" "prediction"] columns))
                                        [{:label 5.0
                                          :prediction 4.9}])
                  log/info (fn [_])]

      (let [res (glr/execute :dataset
                             {:model_path "/tmp/model"
                              :feature_field ["feature1"]})]

        (is (= {:data [{:label 5.0 :prediction 4.9}]} res))))))

(deftest execute-negative-and-edge
  (testing "Transform failure throws exception"
    (with-redefs [glr/transform (fn [_ _ _]
                                  (throw (RuntimeException.
                                          "transform failed")))

                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"GeneralizedLinearRegressionModel execution failed"

           (glr/execute :dataset {:model_path "/tmp/model"
                                  :feature_field ["feature1"]})))))

  (testing "dataset->json failure throws exception"
    (with-redefs [glr/transform (fn [_ _ _]
                                  :transformed-dataset)

                  utils/dataset->json (fn [_ _]
                                        (throw (RuntimeException.
                                                "json conversion failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"GeneralizedLinearRegressionModel execution failed"
           (glr/execute :dataset
                        {:model_path "/tmp/model"
                         :feature_field ["feature1"]}))))))
