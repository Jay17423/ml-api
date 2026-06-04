(ns ml-api.preprocessing-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.preprocessing :as preprocess]))

(deftest normalize-columns-test
  (testing "Returns vector for string input"
    (is (= ["salary"] (preprocess/normalize-columns "salary"))))

  (testing "Returns same vector for vector input"
    (is (= ["salary" "bonus"] (preprocess/normalize-columns
                               ["salary" "bonus"]))))

  (testing "Returns empty vector for invalid input"
    (is (= [] (preprocess/normalize-columns nil)))))

(deftest get-required-columns-test
  (testing "Extracts all required columns"
    (is (= ["salary" "bonus" "review_text" "target"]
           (preprocess/get-required-columns {:feature_field ["salary" "bonus"]
                                             :text_field "review_text"
                                             :target_field "target"}))))

  (testing "Removes duplicate columns"
    (is (= ["salary"] (preprocess/get-required-columns
                       {:feature_field ["salary"]
                        :input_field "salary"}))))

  (testing "Returns empty vector when params empty"
    (is (= [] (preprocess/get-required-columns {})))))

(deftest preprocess-dataset-negative-test
  (testing "Throws preprocessing exception"
    (with-redefs [preprocess/get-required-columns
                  (fn [_]
                    (throw (RuntimeException. "failed")))]

      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo
           #"Dataset preprocessing failed"
           (preprocess/preprocess-dataset
            :dataset
            {}))))))