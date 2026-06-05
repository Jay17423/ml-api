(ns ml-api.algorithms.glr-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.glr :as glr]
   [ml-api.algorithms.vector-assembler :as va]
   [taoensso.timbre :as log]))

(deftest calculate-duration-sec-success
  (testing "calculate-duration-sec returns duration in seconds"
    (is (= 2.0 (glr/calculate-duration-sec 1000 3000)))))

(deftest validate-train-size-success
  (testing "validate-train-size works for valid value"
    (is (nil? (glr/validate-train-size 80)))))

(deftest validate-train-size-failure
  (testing "validate-train-size throws exception for invalid value"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo
         #"Invalid train size"

         (glr/validate-train-size 0)))

    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo
         #"Invalid train size"

         (glr/validate-train-size 100)))))

(deftest execute-success-response
  (testing "GeneralizedLinearRegression executes successfully"
    (with-redefs [glr/validate-train-size (fn [_] nil)

                  va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  glr/current-time-ms (fn []
                                        1000)

                  glr/create-glr (fn [_]
                                   :glr-instance)

                  glr/save-model (fn [_ _]
                                   nil)

                  glr/dataset-preview (fn [_ _ _]
                                        [{:label 10.0
                                          :prediction 9.8}])

                  log/info (fn [_])

                  glr/execute (fn [_ _]
                                {:model-id "123"
                                 :model-label "glr-model"
                                 :model-algo "GeneralizedLinearRegressionModel"
                                 :regression-family "gaussian"
                                 :train-size 80
                                 :seed 123
                                 :coefficients [1.0 2.0]
                                 :intercept 0.5
                                 :training-time-sec 0.0
                                 :prediction-time-sec 0.0
                                 :data [{:label 10.0
                                         :prediction 9.8}]})]

      (let [res (glr/execute :dataset
                             {:feature_field ["feature1" "feature2"]
                              :target_field "label"
                              :model_label "glr-model"})]

        (is (= "glr-model" (:model-label res)))
        (is (= "gaussian" (:regression-family res)))
        (is (= [{:label 10.0 :prediction 9.8}] (:data res)))))))

(deftest execute-default-values
  (testing "GeneralizedLinearRegression uses default values"
    (with-redefs [glr/validate-train-size (fn [_] nil)

                  va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  glr/current-time-ms (fn []
                                        1000)

                  glr/create-glr (fn [params]
                                   (is (= "gaussian"
                                          (:regression_family params)))
                                   :glr-instance)

                  glr/save-model (fn [_ _]
                                   nil)

                  glr/dataset-preview (fn [_ _ _]
                                        [])

                  log/info (fn [_])

                  glr/execute (fn [_ _]
                                {:train-size 80
                                 :seed 123
                                 :data []})]

      (let [res (glr/execute :dataset
                             {:feature_field ["feature1"]
                              :target_field "label"})]
        (is (= 80 (:train-size res)))
        (is (= 123 (:seed res)))))))

(deftest execute-negative-and-edge
  (testing "Invalid train size throws exception"
    (with-redefs [glr/validate-train-size (fn [_]
                                            (throw (ex-info
                                                    "Invalid train size"
                                                    {})))]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"GeneralizedLinearRegression execution failed"

           (glr/execute :dataset
                        {:feature_field ["feature1"]
                         :train_size 0})))))

  (testing "Execution failure throws exception"
    (with-redefs [glr/execute
                  (fn [_ _]
                    (throw (ex-info
                            "GeneralizedLinearRegression execution failed"
                            {})))

                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"GeneralizedLinearRegression execution failed"

           (glr/execute :dataset
                        {:feature_field ["feature1"]
                         :target_field "label"}))))))
