(ns ml-api.core-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.core :as core]
   [ml-api.services.dataset :as ds]
   [ml-api.preprocessing :as preprocess]
   [ml-api.specs :as specs]
   [ml-api.algorithms.tokenizer :as tokenizer]
   [taoensso.timbre :as log]))

(deftest execute-algorithm-success
  (testing "Tokenizer algorithm executes successfully"
    (with-redefs [tokenizer/execute-tokenizer
                  (fn [_ _]
                    {:data [{:tokens ["hello"]}]})]
      (is (= {:data [{:tokens ["hello"]}]}
             (core/execute-algorithm "Tokenizer"
                                     :dataset {:input_field "text"})))))

  (testing "Unsupported algorithm throws exception"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo
         #"Unsupported algorithm"
         (core/execute-algorithm "UnknownAlgo" :dataset {})))))

(deftest execute-success-response
  (with-redefs [specs/validate-request identity
                ds/read-dataset (fn [_ _]
                                  :raw-dataset)

                preprocess/preprocess-dataset (fn [_ _]
                                                :processed-dataset)

                core/execute-algorithm (fn [_ _ _]
                                         {:data [{:prediction 100.0}]})

                log/info (fn [_])]

    (let [res (core/execute
               {:body {:algorithm "Tokenizer"
                       :parameters {:input_field "text"}}
                :start-time 0})]
      (is (= 200 (:status res)))
      (is (= "success" (get-in res [:body :status])))
      (is (= {:data [{:prediction 100.0}]} (get-in res [:body :result])))
      (is (= {:input_field "text"} (get-in res [:body :parameters]))))))

(deftest execute-negative-and-edge
  (testing "Invalid request throws validation exception"
    (with-redefs [specs/validate-request
                  (fn [_]
                    (throw (AssertionError. "invalid request")))]

      (is (thrown-with-msg?
           AssertionError
           #"invalid request"

           (core/execute {:body {}
                          :start-time 0})))))

  (testing "Dataset read failure throws exception"
    (with-redefs [specs/validate-request identity
                  ds/read-dataset
                  (fn [_ _]
                    (throw (RuntimeException. "dataset failed")))
                  log/info
                  (fn [_])]

      (is (thrown-with-msg?
           RuntimeException
           #"dataset failed"

           (core/execute
            {:body {:algorithm "Tokenizer"
                    :parameters {}}
             :start-time 0})))))

  (testing "Algorithm execution failure throws exception"
    (with-redefs [specs/validate-request identity
                  ds/read-dataset (fn [_ _]
                                    :raw-dataset)

                  preprocess/preprocess-dataset (fn [_ _]
                                                  :processed-dataset)

                  core/execute-algorithm
                  (fn [_ _ _]
                    (throw (RuntimeException. "execution failed")))

                  log/info (fn [_])]

      (is (thrown-with-msg?
           RuntimeException
           #"execution failed"
           (core/execute
            {:body {:algorithm "Tokenizer"
                    :parameters {}}
             :start-time 0}))))))