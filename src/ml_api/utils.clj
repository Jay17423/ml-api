(ns ml-api.utils
  (:require
   [flambo.sql :as sql])
  (:import
   [org.apache.spark.ml.linalg DenseVector SparseVector]))

(defn vector->clojure
  "Converts Spark vectors into normal Clojure vectors."
  [value] 
  (cond
    ;; Dense Vector
    (instance? DenseVector value) 
    (vec (.toArray value))
    ;; Sparse Vector
    (instance? SparseVector value)
    (vec (.toArray value))
    ;; Default
    :else
    value))

(defn normalize-row
  "Normalizes Spark row map values."
  [row-map]
  (reduce-kv
   (fn [acc k v]
     (assoc acc k (vector->clojure v)))
   {}
   row-map))

(defn dataset->preview
  "Converts Spark dataset preview into JSON-safe maps."
  [dataset]
  (->> (.limit dataset 10)
       sql/collect
       (map sql/row->map)
       (map normalize-row)))