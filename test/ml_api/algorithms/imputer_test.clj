(ns ml-api.algorithms.imputer-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.imputer :as imp]
   [ml-api.utils :as utils]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "Imputer executes successfully"
    (with-redefs [imp/transform (fn [_ _ _ _ _ _]
                                  :transformed-dataset)

                  utils/preview-columns (fn [_ _]
                                          [:feature :imputed-feature])

                  utils/dataset->json (fn [_ _]
                                        [{:feature 10.0
                                          :imputed-feature 11.0}])

                  log/info (fn [_])]

      (let [res (imp/execute :dataset
                             {:feature_field ["feature"]
                              :output_field ["imputed-feature"]
                              :strategy "mean"
                              :missing_value ["NaN"]
                              :relative_error 0.001})]

        (is (= {:data [{:feature 10.0 :imputed-feature 11.0}]} res))))))

(deftest execute-default-values
  (testing "Imputer uses default values"
    (with-redefs [imp/transform
                  (fn [_ _ _ strategy missing-value relative-error]
                    (is (= "mean" strategy))
                    (is (= ["NaN"] missing-value))
                    (is (= 0.0001 relative-error))
                    :transformed-dataset)

                  utils/preview-columns (fn [_ _]
                                          [:feature :imputed-feature])

                  utils/dataset->json (fn [_ _]
                                        [{:feature 5.0
                                          :imputed-feature 5.5}])

                  log/info (fn [_])]

      (let [res (imp/execute :dataset
                             {:feature_field ["feature"]
                              :output_field ["imputed-feature"]})]

        (is (= {:data [{:feature 5.0 :imputed-feature 5.5}]} res))))))

(deftest execute-negative-and-edge
  (testing "Transform failure throws exception"
    (with-redefs [imp/transform (fn [_ _ _ _ _ _]
                                  (throw (RuntimeException.
                                          "transform failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Imputer execution failed"

           (imp/execute :dataset
                        {:feature_field ["feature"]
                         :output_field ["imputed-feature"]})))))

  (testing "dataset->json failure throws exception"
    (with-redefs [imp/transform (fn [_ _ _ _ _ _]
                                  :transformed-dataset)

                  utils/preview-columns (fn [_ _]
                                          [:feature :imputed-feature])

                  utils/dataset->json (fn [_ _]
                                        (throw (RuntimeException.
                                                "json conversion failed")))
                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Imputer execution failed"
           
           (imp/execute :dataset
            {:feature_field ["feature"]
             :output_field ["imputed-feature"]}))))))
