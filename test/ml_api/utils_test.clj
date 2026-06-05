(ns ml-api.utils-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [ml-api.utils :as utils])
  (:import
   [org.apache.spark.ml.linalg Vectors]
   [scala.collection.mutable WrappedArray]))

(deftest vector->clojure-test
  (testing "Converts Spark vector into Clojure vector"
    (is (= [1.0 2.0 3.0] (utils/vector->clojure
                          (Vectors/dense (double-array [1.0 2.0 3.0])))))))

(deftest wrapped-array->clojure-test
  (testing "Converts WrappedArray into Clojure vector"
    (let [wrapped-array (WrappedArray/make (into-array String ["a" "b" "c"]))]
      (is (= ["a" "b" "c"] (utils/wrapped-array->clojure wrapped-array))))))

(deftest normalize-value-test
  (testing "Normalizes Spark vector"
    (is (= [1.0 2.0] (utils/normalize-value
                      (Vectors/dense (double-array [1.0 2.0]))))))

  (testing "Normalizes WrappedArray"
    (let [wrapped-array (WrappedArray/make (into-array String ["x" "y"]))]
      (is (= ["x" "y"] (utils/normalize-value wrapped-array)))))

  (testing "Returns normal value as it is"
    (is (= "hello" (utils/normalize-value "hello")))))

(deftest preview-columns-test
  (testing "Combines multiple field collections"
    (is (= ["salary" "bonus" "prediction"]
           (utils/preview-columns ["salary" "bonus"] ["prediction"]))))

  (testing "Returns empty vector for no inputs"
    (is (= [] (utils/preview-columns)))))

(deftest normalize-row-test
  (testing "Normalizes Spark-specific values inside map"
    (let [wrapped-array (WrappedArray/make (into-array String ["a" "b"]))]
      (is (= {:features [1.0 2.0]
              :tokens ["a" "b"]
              :name "jay"}
             (utils/normalize-row {:features
                                   (Vectors/dense (double-array [1.0 2.0]))
                                   :tokens wrapped-array
                                   :name "jay"})))))

  (testing "Returns same map for normal values"
    (is (= {:age 25} (utils/normalize-row {:age 25})))))