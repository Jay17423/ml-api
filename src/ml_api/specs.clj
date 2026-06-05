(ns ml-api.specs
  "Request validation specifications for ML algorithms."
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
   {:required {:url {:validator string?
                     :type "string"}

               :feature_field {:validator #(and (vector? %) (every? string? %))
                               :type "vector<string>"}

               :label_field {:validator string?
                             :type "string"}}

    :optional {:flatten {:validator boolean?
                         :type "boolean"}}}

   "CountVectorizer"
   {:required {:url {:validator string?
                     :type "string"}

               :input_field {:validator string?
                             :type "string"}

               :output_field {:validator string?
                              :type "string"}}

    :optional {:word_limit {:validator integer?
                            :type "integer"}

               :min_docs {:validator integer?
                          :type "integer"}

               :min_count {:validator float?
                           :type "float"}}}

   "Tokenizer"
   {:required {:url {:validator string?
                     :type "string"}

               :input_field {:validator string?
                             :type "string"}

               :output_field {:validator string?
                              :type "string"}}

    :optional {}}

   "RegexTokenizer"
   {:required {:url {:validator string?
                     :type "string"}

               :input_field {:validator string?
                             :type "string"}

               :output_field {:validator string?
                              :type "string"}}

    :optional {:pattern {:validator string?
                         :type "string"}

               :pattern_as_delimiter {:validator boolean?
                                      :type "boolean"}

               :minimum_token_length {:validator integer?
                                      :type "integer"}

               :convert_to_lowercase {:validator boolean?
                                      :type "boolean"}}}

   "StopWordsRemover"
   {:required {:url {:validator string?
                     :type "string"}

               :input_field {:validator string?
                             :type "string"}

               :output_field {:validator string?
                              :type "string"}}

    :optional {:stop_words {:validator #(and (vector? %) (every? string? %))
                            :type "vector<string>"}
               :case_sensitive {:validator boolean?
                                :type "boolean"}}}

   "NGram"
   {:required {:url {:validator string?
                     :type "string"}

               :input_field {:validator string?
                             :type "string"}

               :output_field {:validator string?
                              :type "string"}}

    :optional {:ngram_size {:validator integer?
                            :type "integer"}}}
   "StringIndexer"
   {:required {:url {:validator string?
                     :type "string"}

               :input_fields {:validator #(and (vector? %) (every? string? %))
                              :type "vector<string>"}

               :output_fields {:validator
                               #(and (vector? %) (every? string? %))
                               :type "vector<string>"}}

    :optional {:invalid_value {:validator string?
                               :type "string"}

               :label_order {:validator string?
                             :type "string"}}}

   "Binarizer"
   {:required {:url {:validator string?
                     :type "string"}

               :input_field {:validator #(and (vector? %) (every? string? %))
                             :type "vector<string>"}

               :output_field {:validator #(and (vector? %) (every? string? %))
                              :type "vector<string>"}}

    :optional {:threshold_values {:validator #(and
                                               (vector? %)
                                               (every? number? %))
                                  :type "vector<float>"}}}

   "Normalizer"
   {:required
    {:url {:validator string?
           :type "string"}

     :feature_field {:validator #(and (vector? %) (every? string? %))
                     :type "vector<string>"}

     :output_field {:validator string?
                    :type "string"}}

    :optional {:norm_value {:validator #(or (number? %) (= % "inf"))
                            :type "float | \"inf\""}}}

   "StandardScaler"
   {:required {:url {:validator string?
                     :type "string"}

               :feature_field {:validator #(and (vector? %) (every? string? %))
                               :type "vector<string>"}
               :output_field {:validator string? :type "string"}}

    :optional {:with_std {:validator boolean?
                          :type "boolean"}

               :with_mean {:validator boolean?
                           :type "boolean"}}}
   "Bucketizer"
   {:required {:url {:validator string?
                     :type "string"}

               :feature_field {:validator #(and (vector? %) (every? string? %))
                               :type "vector<string>"}

               :output_field {:validator #(and (vector? %) (every? string? %))
                              :type "vector<string>"}

               :bucket_splits {:validator
                               #(and
                                 (vector? %)
                                 (every? (fn [x] (and (vector? x)
                                                      (every? number? x)))
                                         %))
                               :type "vector<vector<float>>"}}

    :optional {:invalid_value {:validator string?
                               :type "string"}}}

   "Imputer"
   {:required {:url {:validator string?
                     :type "string"}

               :feature_field {:validator #(and (vector? %) (every? string? %))
                               :type "vector<string>"}

               :output_field {:validator #(and (vector? %) (every? string? %))
                              :type "vector<string>"}}

    :optional {:strategy {:validator #(contains? #{"mean" "median" "mode"} %)
                          :type "mean | median | mode"}

               :missing_value {:validator vector?
                               :type "vector<string|float>"}

               :relative_error {:validator number?
                                :type "float"}}}
   "ChiSqSelector"
   {:required {:url {:validator string?
                     :type "string"}

               :feature_field {:validator #(and (vector? %) (every? string? %))
                               :type "vector<string>"}

               :output_field {:validator string?
                              :type "string"}}

    :optional {:target_field {:validator string?
                              :type "string"}

               :selection_method
               {:validator
                #(contains? #{"numTopFeatures"
                              "percentile"
                              "fpr"
                              "fdr"
                              "fwe"}
                            %)
                :type "numTopFeatures | percentile | fpr | fdr | fwe"}

               :top_feature {:validator int?
                             :type "integer"}

               :selection_percentage {:validator number?
                                      :type "float"}

               :fpr_threshold {:validator number?
                               :type "float"}

               :fdr_threshold {:validator number?
                               :type "float"}

               :fwe_threshold {:validator number?
                               :type "float"}}}

   "BucketedRandomProjectionLSH"
   {:required {:url {:validator string?
                     :type "string"}

               :feature_field {:validator
                               #(and (vector? %) (every? string? %))
                               :type "vector<string>"}

               :output_field {:validator string?
                              :type "string"}

               :bucket_length {:validator number?
                               :type "float"}}

    :optional {:num_hash_tables {:validator int?
                                 :type "integer"}

               :random_seed {:validator int?
                             :type "integer"}}}

   "GeneralizedLinearRegression"
   {:required {:url {:validator string?
                     :type "string"}

               :feature_field {:validator
                               #(and (vector? %) (every? string? %))
                               :type "vector<string>"}

               :target_field {:validator string?
                              :type "string"}}

    :optional {:output_field {:validator string?
                              :type "string"}

               :model_path {:validator string?
                            :type "string"}

               :model_label {:validator string?
                             :type "string"}

               :train_size {:validator
                            #(and (int? %) (> % 0) (< % 100))
                            :type "integer(1-99)"}

               :seed {:validator int?
                      :type "integer"}

               :regression_family
               {:validator
                #(contains? #{"gaussian" "binomial" "poisson" "gamma" "tweedie"}
                            (.toLowerCase %))
                :type "gaussian | binomial | poisson | gamma | tweedie"}

               :prediction_link {:validator string?
                                 :type "string"}

               :include_intercept {:validator boolean?
                                   :type "boolean"}

               :max_iterations {:validator int?
                                :type "integer"}

               :training_tolerance {:validator number?
                                    :type "float"}

               :regularization_strength {:validator number?
                                         :type "float"}

               :row_weight_field {:validator string?
                                  :type "string"}

               :training_solver {:validator string?
                                 :type "string"}

               :link_output_field {:validator string?
                                   :type "string"}

               :offset_field {:validator string?
                              :type "string"}

               :aggregation_depth {:validator int?
                                   :type "integer"}}}
   "LDA"
   {:required {:url {:validator string?
                     :type "string"}

               :text_field {:validator string?
                            :type "string"}}

    :optional {:topic_count {:validator #(and (int? %) (>= % 1))
                             :type "integer >= 1"}

               :max_iterations {:validator #(and (int? %) (>= % 1))
                                :type "integer >= 1"}

               :random_seed {:validator #(and (int? %) (>= % 1))
                             :type "integer >= 1"}

               :training_method {:validator
                                 #(contains? #{"online" "em"} (.toLowerCase %))
                                 :type "online | em"}

               :checkpoint_interval {:validator #(and (int? %) (>= % 1))
                                     :type "integer >= 1"}

               :initial_learning_offset {:validator #(and (number? %) (> % 0))
                                         :type "float > 0"}

               :learning_decay_rate {:validator
                                     #(and (number? %) (> % 0.5) (<= % 1.0))
                                     :type "float (0.5,1.0]"}

               :training_sample_rate {:validator
                                      #(and (number? %) (>= % 0.0) (<= % 1.0))
                                      :type "float [0,1]"}

               :auto_optimize {:validator boolean?
                               :type "boolean"}

               :document_concentration {:validator
                                        #(and (vector? %) (every? number? %))
                                        :type "vector<float>"}

               :topic_concentration {:validator #(and (number? %) (> % 0))
                                     :type "float > 0"}

               :topic_distribution_field {:validator string?
                                          :type "string"}

               :keep_last_checkpoint {:validator boolean?
                                      :type "boolean"}}}

   "GeneralizedLinearRegressionModel"
   {:required {:url {:validator string?
                     :type "string"}

               :model_path {:validator string?
                            :type "string"}

               :feature_field {:validator
                               #(and (vector? %) (every? string? %))
                               :type "vector<string>"}

               :target_field {:validator string?
                              :type "string"}}

    :optional {:output_field {:validator string?
                              :type "string"}}}})

(defn validate-parameter-type
  [param-name validator expected-type value]

  (when-not (validator value)
    (throw (ex-info "Invalid parameter datatype"
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

  (let [algorithm (:algorithm body)
        parameters (:parameters body)
        algorithm-spec (get algorithm-params algorithm)]

    (when-not algorithm-spec
      (throw (ex-info "Unsupported algorithm"
                      {:type :validation/unsupported-algorithm
                       :algorithm algorithm})))
    (validate-required-parameters (:required algorithm-spec) parameters)
    (validate-optional-parameters (:optional algorithm-spec) parameters)))