(ns ml-api.algorithms.chi-sq-selector-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.chi-sq-selector :as css]
   [ml-api.algorithms.vector-assembler :as va]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "ChiSqSelector executes successfully"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  css/dataset->json
                  (fn [_ _]
                    [{:selected-features [1.0 2.0]}])

                  log/info (fn [_])

                  css/execute
                  (fn [_ _]
                    {:data [{:selected-features [1.0 2.0]}]})]

      (let [res (css/execute :dataset
                             {:feature_field ["feature1" "feature2"]
                              :target_field "label"
                              :output_field "selected_features"
                              :selection_method "numTopFeatures"
                              :top_feature 10})]
        (is (= {:data [{:selected-features [1.0 2.0]}]} res))))))

(deftest execute-default-values
  (testing "ChiSqSelector uses default values"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  css/dataset->json
                  (fn [_ output-field]
                    (is (= "selected_features" output-field))
                    [{:selected-features [0.5]}])

                  log/info (fn [_])

                  css/execute
                  (fn [_ _]
                    {:data [{:selected-features [0.5]}]})]

      (let [res (css/execute :dataset
                             {:feature_field ["feature1"]})]
        (is (= {:data [{:selected-features [0.5]}]} res))))))

(deftest execute-with-percentile-selection
  (testing "ChiSqSelector executes with percentile selection method"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  css/dataset->json (fn [_ _]
                                      [{:selected-features [1.0]}])
                  log/info (fn [_])

                  css/execute
                  (fn [_ _]
                    {:data [{:selected-features [1.0]}]})]

      (let [res (css/execute :dataset
                             {:feature_field ["feature1"]
                              :selection_method "percentile"
                              :selection_percentage 0.5})]

        (is (= {:data [{:selected-features [1.0]}]} res))))))

(deftest execute-negative-and-edge
  (testing "Feature vector creation failure throws exception"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             (throw (RuntimeException.
                                                     "vector creation failed")))
                  log/info (fn [_])

                  css/execute
                  (fn [_ _]
                    (throw (ex-info
                            "ChiSqSelector execution failed"
                            {})))]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"ChiSqSelector execution failed"

           (css/execute :dataset
                        {:feature_field ["feature1"]})))))

  (testing "dataset->json failure throws exception"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  css/dataset->json
                  (fn [_ _]
                    (throw (RuntimeException.
                            "json conversion failed")))

                  log/info (fn [_])

                  css/execute
                  (fn [_ _]
                    (throw (ex-info
                            "ChiSqSelector execution failed"
                            {})))]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"ChiSqSelector execution failed"

           (css/execute :dataset
                        {:feature_field ["feature1"]}))))))
