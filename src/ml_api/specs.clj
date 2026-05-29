(ns ml-api.specs
  (:require
   [clojure.spec.alpha :as s]))

;; ------------------------------------------------------------
;; Primitive Specs
;; ------------------------------------------------------------

(s/def ::string string?)

(s/def ::boolean boolean?)

(s/def ::string-vector
  (s/coll-of string?
             :kind vector?
             :min-count 1))

;; ------------------------------------------------------------
;; Algorithm Parameter Definitions
;; ------------------------------------------------------------

(def algorithm-params

  {"ChiSquareTest"

   {:required
    {:url string?
     :feature_field vector?
     :label_field string?}

    :optional
    {:flatten boolean?}}})

;; ------------------------------------------------------------
;; Generic Validation
;; ------------------------------------------------------------

(defn validate-parameter-type
  [param-name expected-type value]

  (when-not (expected-type value)

    (throw
     (ex-info
      "Invalid parameter datatype"

      {:type :validation/invalid-parameter-type
       :parameter param-name
       :expected (str expected-type)
       :received (type value)}))))

(defn validate-required-parameters
  [required-spec parameters]

  (doseq [[param-name validator] required-spec]

    (when-not (contains? parameters param-name)

      (throw
       (ex-info
        "Missing required parameter"

        {:type :validation/missing-parameter
         :parameter param-name})))

    (validate-parameter-type
     param-name
     validator
     (get parameters param-name))))

(defn validate-optional-parameters
  [optional-spec parameters]

  (doseq [[param-name validator] optional-spec]

    (when (contains? parameters param-name)

      (validate-parameter-type
       param-name
       validator
       (get parameters param-name)))))

(defn validate-request
  [body]

  ;; ----------------------------------------------------------
  ;; Top Level Validation
  ;; ----------------------------------------------------------

  (when-not (string? (:algorithm body))

    (throw
     (ex-info
      "algorithm must be string"

      {:type :validation/invalid-algorithm})))

  (when-not (map? (:parameters body))

    (throw
     (ex-info
      "parameters must be object"

      {:type :validation/invalid-parameters})))

  ;; ----------------------------------------------------------
  ;; Algorithm Validation
  ;; ----------------------------------------------------------

  (let [algorithm (:algorithm body)

        parameters (:parameters body)

        algorithm-spec
        (get algorithm-params algorithm)]

    (when-not algorithm-spec

      (throw
       (ex-info
        "Unsupported algorithm"

        {:type :validation/unsupported-algorithm
         :algorithm algorithm})))

    ;; --------------------------------------------------------
    ;; Required Params
    ;; --------------------------------------------------------

    (validate-required-parameters
     (:required algorithm-spec)
     parameters)

    ;; --------------------------------------------------------
    ;; Optional Params
    ;; --------------------------------------------------------

    (validate-optional-parameters
     (:optional algorithm-spec)
     parameters)))