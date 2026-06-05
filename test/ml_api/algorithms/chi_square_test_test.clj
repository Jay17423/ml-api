(ns ml-api.algorithms.chi-square-test-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.chi-square-test :as cht]
   [ml-api.algorithms.vector-assembler :as va]
   [taoensso.timbre :as log]))

(deftest execute-success-response
  (testing "ChiSquareTest executes successfully"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  cht/parse-result (fn [_ _]
                                                 {:p-values [0.1]
                                                  :degrees-of-freedom [1]
                                                  :statistics [5.0]})
                  log/info (fn [_])

                  cht/execute (fn [_ _]
                                            {:data {:p-values [0.1]
                                                    :degrees-of-freedom [1]
                                                    :statistics [5.0]}})]
      (let [res (cht/execute :dataset
                                         {:label_field "label"
                                          :feature_field ["feature1"]
                                          :flatten true})]
        (is (= {:data {:p-values [0.1] :degrees-of-freedom [1]
                       :statistics [5.0]}}
               res))))))

(deftest execute-default-flatten
  (testing "ChiSquareTest uses default flatten value"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  cht/parse-result (fn [_ flatten]
                                                 (is (= true flatten))
                                                 {:p-values [0.2]})
                  log/info (fn [_])

                  cht/execute (fn [_ _]
                                            {:data {:p-values [0.2]}})]

      (let [res (cht/execute :dataset
                                         {:label_field "label"
                                          :feature_field ["feature1"]})]
        (is (= {:data {:p-values [0.2]}} res))))))

(deftest execute-negative-and-edge
  (testing "Feature vector creation failure throws exception"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             (throw (RuntimeException.
                                                     "vector creation failed")))
                  cht/execute
                  (fn [_ _]
                    (throw (ex-info
                            "ChiSquareTest execution failed"
                            {})))]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"ChiSquareTest execution failed"

           (cht/execute :dataset
                                    {:label_field "label"
                                     :feature_field ["feature1"]})))))

  (testing "parse-result failure throws exception"
    (with-redefs [va/create-feature-vector (fn [_ _]
                                             :vectorized-dataset)

                  cht/parse-result (fn [_ _]
                                                 (throw (RuntimeException.
                                                         "parse failed")))
                  cht/execute
                  (fn [_ _]
                    (throw (ex-info
                            "ChiSquareTest execution failed"
                            {})))]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"ChiSquareTest execution failed"
           (cht/execute :dataset
                                    {:label_field "label"
                                     :feature_field ["feature1"]}))))))
