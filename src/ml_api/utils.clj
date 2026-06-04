(ns ml-api.utils
  (:require
   [flambo.sql :as sql])
  (:import
   [org.apache.spark.ml.linalg Vector]
   [scala.collection.mutable WrappedArray]))

(defn vector->clojure
  "Converts Spark vectors into normal Clojure vectors."
  [value]
  (cond
    (instance? Vector value)
    (vec (.toArray ^Vector value))
    :else
    value))

(defn wrapped-array->clojure
  "Converts Scala WrappedArray into Clojure vector."
  [wrapped-array]
  (vec (.array ^WrappedArray wrapped-array)))

(defn normalize-row
  "Normalizes Spark row map values."
  [row-map]
  (reduce-kv
   (fn [acc k v]
     (assoc acc k (vector->clojure v)))
   {}
   row-map))

(comment (defn dataset->preview
  "Converts Spark dataset preview into JSON-safe maps."
  [ds]
  (->> (.limit ds 10)
       sql/collect
       (map sql/row->map)
       (map normalize-row))))