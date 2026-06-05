(ns ml-api.algorithms.vector-assembler-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.algorithms.vector-assembler :as va]
   [taoensso.timbre :as log]))

(deftest create-feature-vector-success
  (testing "Feature vector created successfully"
    (with-redefs [va/create-feature-vector (fn [_ feature-cols]
                                             (is (= ["feature1" "feature2"]
                                                    feature-cols))
                                             :vectorized-dataset)
                  log/info (fn [_])]

      (let [res (va/create-feature-vector :dataset ["feature1" "feature2"])]
        (is (= :vectorized-dataset res))))))

(deftest create-feature-vector-negative-and-edge
  (testing "Feature vector creation failure throws exception"
    (with-redefs [va/create-feature-vector
                  (fn [_ _]
                    (throw (ex-info
                            "Feature vector creation failed"
                            {})))

                  log/info (fn [_])]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Feature vector creation failed"
           (va/create-feature-vector :dataset ["feature1" "feature2"]))))))
