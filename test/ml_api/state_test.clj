(ns ml-api.state-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.state :as state]
   [flambo.session :as fs]
   [flambo.sql :as sql]
   [omniconf.core :as cfg]
   [taoensso.timbre :as log]))

(deftest create-session-test
  (testing "Creates Spark session successfully"
    (with-redefs [fs/session-builder (fn []
                                       :builder)
                  fs/master (fn [_ _]
                              :builder)

                  fs/app-name (fn [_ _]
                                :builder)

                  fs/get-or-create (fn [_]
                                     :spark-session)

                  cfg/get (fn [& _]
                            "local[*]")]
      (is (= :spark-session (state/create-session)))))

  (testing "Throws exception when session creation fails"
    (with-redefs [fs/session-builder (fn []
                                       (throw (RuntimeException.
                                               "spark failed")))
                  cfg/get (fn [& _]
                          "local[*]")]

      (is (thrown-with-msg? clojure.lang.ExceptionInfo
                            #"Unable to create Spark session"

                            (state/create-session))))))

(deftest warmup-spark-test
  (testing "Warms up Spark successfully"
    (with-redefs [sql/create-dataset (fn [_ _]
                                       :dataset)

                  sql/count (fn [_]
                              4)

                  log/info (fn [_])]

      (is (= :spark (state/warmup-spark :spark)))))

  (testing "Throws exception when warmup fails"
    (with-redefs [sql/create-dataset
                  (fn [_ _]
                    (throw (RuntimeException.
                            "warmup failed")))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo
                            #"Spark warmup failed"

                            (state/warmup-spark
                             :spark))))))