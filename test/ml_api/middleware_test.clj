(ns ml-api.middleware-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.middleware :as mw]
   [taoensso.timbre :as log]))

(deftest error-response-test
  (let [res (mw/error-response 400 "Validation failed" 100 {:field "algorithm"})]
    (is (= 400 (:status res)))
    (is (= "error" (get-in res [:body :status])))
    (is (= "Validation failed" (get-in res [:body :message])))
    (is (= {:field "algorithm"} (get-in res [:body :details])))
    (is (= 100 (get-in res [:body :duration-ms])))))

(deftest status-for-error-test
  (testing "Returns correct status for known errors"
    (is (= 404 (mw/status-for-error :validation/invalid-algorithm)))
    (is (= 400 (mw/status-for-error :validation/missing-parameter)))
    (is (= 500 (mw/status-for-error :unknown/error))))
  (testing "Returns 500 for nil error type"
    (is (= 500  (mw/status-for-error nil)))))

(deftest wrap-error-handler-success
  (with-redefs [log/info (fn [_])]
    (let [handler (mw/wrap-error-handler (fn [_]
                                           {:status 200
                                            :body {:status "success"}}))
          res (handler {:start-time 0})]
      (is (= 200 (:status res)))
      (is (= "success" (get-in res [:body :status]))))))

(deftest wrap-error-handler-error
  (with-redefs [log/warn (fn [_])]
    (let [handler (mw/wrap-error-handler
                   (fn [_]
                     (throw (AssertionError. "invalid request"))))

          res (handler {:start-time 0})]
      (is (= 400 (:status res)))
      (is (= "Validation failed" (get-in res [:body :message])))
      (is (= "invalid request" (get-in res [:body :details]))))))

(deftest wrap-error-handler-exception
  (with-redefs [log/error (fn [_])]
    (let [handler (mw/wrap-error-handler
                   (fn [_]
                     (throw (ex-info "Missing field"
                                     {:type :validation/missing-parameter
                                      :field "input_field"}))))
          res (handler {:start-time 0})]
      (is (= 400 (:status res)))
      (is (= "Missing required parameter" (get-in res [:body :message])))
      (is (= :validation/missing-parameter (get-in
                                            res [:body :details :type]))))))

(deftest wrap-error-handler-unhandled
  (with-redefs [log/error (fn [_])]
    (let [handler (mw/wrap-error-handler
                   (fn [_]
                     (throw (RuntimeException. "server crashed"))))
          res (handler {:start-time 0})]
      (is (= 500 (:status res)))
      (is (= "Internal server error" (get-in res [:body :message])))
      (is (= "server crashed" (get-in res [:body :details]))))))

(deftest wrap-request-timer-test
  (let [handler (mw/wrap-request-timer
                 (fn [req]
                   req))
        res (handler {:body {:data 1}})]

    (is (contains? res :start-time))
    (is (= {:data 1} (:body res)))
    (is (number? (:start-time res)))))