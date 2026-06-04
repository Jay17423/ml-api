(ns ml-api.utils-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.linalg Vectors]
   [scala.collection.mutable WrappedArray]))

(deftest vector->clojure-test
  (testing "Converts Spark vector into Clojure vector"
    (is (= [1.0 2.0 3.0]
           (utils/vector->clojure (Vectors/dense
                                   (double-array [1.0 2.0 3.0]))))))

  (testing "Returns same value for non-vector input"
    (is (= "hello" (utils/vector->clojure "hello")))))

(deftest wrapped-array->clojure-test
  (testing "Converts WrappedArray into Clojure vector"
    (let [wrapped-array (WrappedArray/make (into-array String ["a" "b" "c"]))]
      (is (= ["a" "b" "c"] (utils/wrapped-array->clojure wrapped-array))))))

(deftest normalize-row-test
  (testing "Normalizes vector values inside row map"
    (is (= {:features [1.0 2.0]
            :name "jay"}
           (utils/normalize-row {:features
                                 (Vectors/dense (double-array [1.0 2.0]))
                                 :name "jay"}))))

  (testing "Returns same values for non-vector fields"
    (is (= {:age 25} (utils/normalize-row {:age 25})))))