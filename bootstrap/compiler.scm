;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;; DMCI: Compiling scheme into composable and
;;       differentiable neural network representations
;;
;; compiler.scm: The self-hosted meta-circular Scheme evaluator that DMCI compiles once into a differentiable interpreter
;;
;; Luke Sheneman
;; Research Computing and Data Services (RCDS)
;; Institute for Interdisciplinary Data Sciences (IIDS)
;; University of Idaho
;; sheneman@uidaho.edu
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; Self-hosted Scheme evaluator -- full bootstrap
;;;
;;; Compiles to a differentiable PyTorch program via the neural_compiler.
;;; Supports self-hosting: can evaluate its own source code, producing
;;; a working evaluator that evaluates Scheme programs.
;;;
;;; Supported: numbers, booleans, symbols, quote, if, cond, let, letrec, lambda,
;;;            define, begin, cons/car/cdr, list, null?/pair?/number?/boolean?/symbol?/eq?/not,
;;;            +/-/*//, =/</>/<=/>=, sin/cos/exp/sqrt/log/abs/pow

;;; --- Environment ---

(define (env-lookup name env)
  (cond
    ((null? env) 0)
    ((eq? (car (car env)) name) (cdr (car env)))
    (#t (env-lookup name (cdr env)))))

(define (env-extend name val env)
  (cons (cons name val) env))

(define (env-extend-many names vals env)
  (if (null? names)
    env
    (env-extend-many
      (cdr names)
      (cdr vals)
      (env-extend (car names) (car vals) env))))

;;; --- Define processing ---

(define (define? form)
  (if (pair? form) (eq? (car form) 'define) #f))

(define (define-name form)
  (let ((target (car (cdr form))))
    (if (pair? target) (car target) target)))

(define (define-value form)
  (let ((target (car (cdr form))))
    (if (pair? target)
      (list 'lambda (cdr target) (car (cdr (cdr form))))
      (car (cdr (cdr form))))))

(define (collect-defines forms)
  (if (null? forms)
    '()
    (if (define? (car forms))
      (cons (cons (define-name (car forms))
                  (define-value (car forms)))
            (collect-defines (cdr forms)))
      (collect-defines (cdr forms)))))

(define (last-form forms)
  (if (null? (cdr forms))
    (car forms)
    (last-form (cdr forms))))

(define (build-defined-env defs env)
  (if (null? defs)
    env
    (let ((name (car (car defs)))
          (val-expr (cdr (car defs))))
      (if (pair? val-expr)
        (if (eq? (car val-expr) 'lambda)
          (build-defined-env (cdr defs)
            (env-extend name
              (list 'defined-fn (car (cdr val-expr)) (car (cdr (cdr val-expr))))
              env))
          (build-defined-env (cdr defs)
            (env-extend name (scheme-eval val-expr env) env)))
        (build-defined-env (cdr defs)
          (env-extend name (scheme-eval val-expr env) env))))))

;;; --- Letrec processing ---
;;; Lexical recursive bindings. Lambda bindings are registered as 'defined-fn
;;; entries (like top-level defines), so a recursive call resolves the binding
;;; through the same mechanism that makes recursive `define` work; non-lambda
;;; bindings are evaluated immediately. All differentiable value flow goes
;;; through the same primitives as the rest of the evaluator, so gradients are
;;; preserved exactly as for recursive `define`.

(define (build-letrec-env bindings env)
  (if (null? bindings)
    env
    (let ((binding (car bindings)))
      (let ((name (car binding))
            (val-expr (car (cdr binding))))
        (if (if (pair? val-expr) (eq? (car val-expr) 'lambda) #f)
          (build-letrec-env (cdr bindings)
            (env-extend name
              (list 'defined-fn (car (cdr val-expr)) (car (cdr (cdr val-expr))))
              env))
          (build-letrec-env (cdr bindings)
            (env-extend name (scheme-eval val-expr env) env)))))))

;;; --- Program evaluation ---

(define (scheme-eval-program forms env)
  (let ((defs (collect-defines forms))
        (body (last-form forms)))
    (let ((prog-env (build-defined-env defs env)))
      (scheme-eval body prog-env))))

;;; --- Evaluator ---

(define (scheme-eval expr env)
  (cond
    ((number? expr) expr)
    ((boolean? expr) expr)
    ((null? expr) expr)
    ((symbol? expr) (env-lookup expr env))
    ((pair? expr)
     (let ((head (car expr)))
       (cond
         ((eq? head 'quote) (car (cdr expr)))
         ((eq? head 'if)
          (let ((test-val (scheme-eval (car (cdr expr)) env)))
            (if test-val
              (scheme-eval (car (cdr (cdr expr))) env)
              (scheme-eval (car (cdr (cdr (cdr expr)))) env))))
         ((eq? head 'cond)
          (eval-cond (cdr expr) env))
         ((eq? head 'let)
          (let ((bindings (car (cdr expr)))
                (body (car (cdr (cdr expr)))))
            (let ((new-env (eval-let-bindings bindings env)))
              (scheme-eval body new-env))))
         ((eq? head 'letrec)
          (let ((bindings (car (cdr expr)))
                (body (car (cdr (cdr expr)))))
            (scheme-eval body (build-letrec-env bindings env))))
         ((eq? head 'lambda)
          (let ((params (car (cdr expr)))
                (body (car (cdr (cdr expr)))))
            (list 'closure params body env)))
         ((eq? head 'begin)
          (eval-begin (cdr expr) env))
         (#t (eval-apply head (cdr expr) env)))))
    (#t 0)))

(define (eval-cond clauses env)
  (if (null? clauses)
    #f
    (let ((clause (car clauses)))
      (if (eq? (car clause) 'else)
        (scheme-eval (car (cdr clause)) env)
        (let ((test-val (scheme-eval (car clause) env)))
          (if test-val
            (scheme-eval (car (cdr clause)) env)
            (eval-cond (cdr clauses) env)))))))

(define (eval-let-bindings bindings env)
  (if (null? bindings)
    env
    (let ((binding (car bindings)))
      (let ((name (car binding))
            (val (scheme-eval (car (cdr binding)) env)))
        (eval-let-bindings (cdr bindings) (env-extend name val env))))))

(define (eval-begin exprs env)
  (if (null? (cdr exprs))
    (scheme-eval (car exprs) env)
    (begin
      (scheme-eval (car exprs) env)
      (eval-begin (cdr exprs) env))))

;;; --- Variadic arithmetic folds ---
;;; (+ a b c ...) and (* ...) fold over all args; (- ...) and (/ ...) are left-
;;; associative with unary forms. These replace the old strictly-binary clauses,
;;; which silently dropped the 3rd and later arguments. All folds are tail-
;;; recursive and use only the host (directly-compiled) primitives + - * /.

(define (sum-rest acc args)
  (if (null? args) acc (sum-rest (+ acc (car args)) (cdr args))))
(define (sum-args args) (sum-rest 0 args))

(define (prod-rest acc args)
  (if (null? args) acc (prod-rest (* acc (car args)) (cdr args))))
(define (prod-args args) (prod-rest 1 args))

(define (sub-rest acc args)
  (if (null? args) acc (sub-rest (- acc (car args)) (cdr args))))
(define (diff-args args)
  (if (null? (cdr args)) (- 0 (car args)) (sub-rest (car args) (cdr args))))

(define (div-rest acc args)
  (if (null? args) acc (div-rest (/ acc (car args)) (cdr args))))
(define (quot-args args)
  (if (null? (cdr args)) (/ 1 (car args)) (div-rest (car args) (cdr args))))

(define (eval-apply func-expr arg-exprs env)
  (let ((func (scheme-eval func-expr env))
        (args (eval-args arg-exprs env)))
    (cond
      ((eq? func-expr '+) (sum-args args))
      ((eq? func-expr '-) (diff-args args))
      ((eq? func-expr '*) (prod-args args))
      ((eq? func-expr '/) (quot-args args))
      ((eq? func-expr '=) (= (car args) (car (cdr args))))
      ((eq? func-expr '<) (< (car args) (car (cdr args))))
      ((eq? func-expr '>) (> (car args) (car (cdr args))))
      ((eq? func-expr '<=) (not (> (car args) (car (cdr args)))))
      ((eq? func-expr '>=) (not (< (car args) (car (cdr args)))))
      ((eq? func-expr 'cons) (cons (car args) (car (cdr args))))
      ((eq? func-expr 'car) (car (car args)))
      ((eq? func-expr 'cdr) (cdr (car args)))
      ((eq? func-expr 'null?) (null? (car args)))
      ((eq? func-expr 'pair?) (pair? (car args)))
      ((eq? func-expr 'number?) (number? (car args)))
      ((eq? func-expr 'boolean?) (boolean? (car args)))
      ((eq? func-expr 'symbol?) (symbol? (car args)))
      ((eq? func-expr 'eq?) (eq? (car args) (car (cdr args))))
      ((eq? func-expr 'not) (not (car args)))
      ((eq? func-expr 'list) args)
      ((eq? func-expr 'sin) (sin (car args)))
      ((eq? func-expr 'cos) (cos (car args)))
      ((eq? func-expr 'exp) (exp (car args)))
      ((eq? func-expr 'sqrt) (sqrt (car args)))
      ((eq? func-expr 'log) (log (car args)))
      ((eq? func-expr 'abs) (abs (car args)))
      ((eq? func-expr 'pow) (pow (car args) (car (cdr args))))
      ((eq? func-expr 'min) (min (car args) (car (cdr args))))
      ((eq? func-expr 'max) (max (car args) (car (cdr args))))
      ((eq? func-expr 'modulo) (modulo (car args) (car (cdr args))))
      ((eq? func-expr 'remainder) (remainder (car args) (car (cdr args))))
      ;; User-defined functions (closures / defined-fns) take precedence over the
      ;; Strategy-B vector ops below, so a program may name a function `scale`/`dot`/etc.
      ;; without it being shadowed by the native op. An unbound vector-op name resolves
      ;; to 0 (not a pair) and falls through to the native clauses.
      ((pair? func)
       (cond
         ((eq? (car func) 'closure)
          (let ((params (car (cdr func)))
                (body (car (cdr (cdr func))))
                (closure-env (car (cdr (cdr (cdr func))))))
            (scheme-eval body (env-extend-many params args closure-env))))
         ((eq? (car func) 'defined-fn)
          (let ((params (car (cdr func)))
                (body (car (cdr (cdr func)))))
            (scheme-eval body (env-extend-many params args env))))
         (#t 0)))
      ;; Strategy B: tensor-payload vector/matrix ops (native; dispatch via VEC_OPS).
      ;; vec/mat take the element list ((vec a b c) is macro-lowered to (vec (list a b c)));
      ;; the rest take vector/matrix refs (and scalars).
      ((eq? func-expr 'vec) (vec (car args)))
      ((eq? func-expr 'mat) (mat (car args)))
      ((eq? func-expr 'ref) (ref (car args) (car (cdr args))))
      ((eq? func-expr 'dot) (dot (car args) (car (cdr args))))
      ((eq? func-expr 'cross) (cross (car args) (car (cdr args))))
      ((eq? func-expr 'norm) (norm (car args)))
      ((eq? func-expr 'normalize) (normalize (car args)))
      ((eq? func-expr 'vsum) (vsum (car args)))
      ((eq? func-expr 'vlen) (vlen (car args)))
      ((eq? func-expr 'scale) (scale (car args) (car (cdr args))))
      ((eq? func-expr 'matvec) (matvec (car args) (car (cdr args))))
      ((eq? func-expr 'matmul) (matmul (car args) (car (cdr args))))
      ((eq? func-expr 'transpose) (transpose (car args)))
      ((eq? func-expr 'trace) (trace (car args)))
      ((eq? func-expr 'det) (det (car args)))
      ((eq? func-expr 'logdet) (logdet (car args)))
      ((eq? func-expr 'inv) (inv (car args)))
      ((eq? func-expr 'outer) (outer (car args) (car (cdr args))))
      ((eq? func-expr 'eye) (eye (car args)))
      ((eq? func-expr 'zeros) (zeros (car args)))
      ((eq? func-expr 'ones) (ones (car args)))
      (#t 0))))

(define (eval-args arg-exprs env)
  (if (null? arg-exprs)
    '()
    (cons (scheme-eval (car arg-exprs) env)
          (eval-args (cdr arg-exprs) env))))
