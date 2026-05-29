(ns ml-api.specs
  (:require
   [clojure.spec.alpha :as s]))

(s/def ::string string?)
(s/def ::boolean boolean?)
(s/def ::string-vector
  (s/coll-of string?
             :kind vector?
             :min-count 1))

(def algorithm-params
  {"ChiSquareTest"
   {:required
    {:url {:validator string?
           :type "string"}

     :feature_field {:validator
                     #(and
                       (vector? %)
                       (every? string? %))
                     :type "vector<string>"}

     :label_field {:validator string?
                   :type "string"}}

    :optional {:flatten
               {:validator boolean?
                :type "boolean"}}}})

(defn validate-parameter-type
  [param-name validator expected-type value]

  (when-not (validator value)
    (throw
     (ex-info
      "Invalid parameter datatype"
      {:type :validation/invalid-parameter-type
       :parameter param-name
       :expected expected-type
       :received (str (type value))}))))

(defn validate-required-parameters
  [required-spec parameters]
  (doseq [[param-name {:keys [validator type]}]
          required-spec]

    (when-not (contains? parameters param-name)
      (throw (ex-info "Missing required parameter"
                      {:type :validation/missing-parameter
                       :parameter param-name})))

    (validate-parameter-type
     param-name validator
     type
     (get parameters param-name))))

(defn validate-optional-parameters
  [optional-spec parameters]

  (doseq [[param-name {:keys [validator type]}] optional-spec]

    (when (contains? parameters param-name)
      (validate-parameter-type
       param-name
       validator
       type
       (get parameters param-name)))))

(defn validate-request
  [body]
  (when-not (string? (:algorithm body))
    (throw (ex-info "algorithm must be string"
                    {:type :validation/invalid-algorithm})))

  (when-not (map? (:parameters body))
    (throw (ex-info "parameters must be object"
                    {:type :validation/invalid-parameters})))

  (let [algorithm
        (:algorithm body)

        parameters
        (:parameters body)

        algorithm-spec
        (get algorithm-params algorithm)]

    (when-not algorithm-spec

      (throw (ex-info "Unsupported algorithm"
                      {:type :validation/unsupported-algorithm
                       :algorithm algorithm})))
    (validate-required-parameters (:required algorithm-spec) parameters)
    (validate-optional-parameters (:optional algorithm-spec) parameters)))