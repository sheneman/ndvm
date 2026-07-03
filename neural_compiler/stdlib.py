############################################################
#
# DMCI: Compiling scheme into composable and
#       differentiable neural network representations
#
# stdlib.py: Scheme standard library — loaded as prelude for compile_program.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Scheme standard library — loaded as prelude for compile_program."""

PRELUDE = """
(define (not x) (if x #f #t))

(define (map f lst)
  (if (null? lst)
    '()
    (cons (f (car lst)) (map f (cdr lst)))))

(define (filter pred lst)
  (if (null? lst)
    '()
    (if (pred (car lst))
      (cons (car lst) (filter pred (cdr lst)))
      (filter pred (cdr lst)))))

(define (fold-left f init lst)
  (if (null? lst)
    init
    (fold-left f (f init (car lst)) (cdr lst))))

(define (fold-right f init lst)
  (if (null? lst)
    init
    (f (car lst) (fold-right f init (cdr lst)))))

(define (for-each f lst)
  (if (null? lst)
    '()
    (begin (f (car lst)) (for-each f (cdr lst)))))

(define (assoc key alist)
  (if (null? alist)
    #f
    (if (equal? key (car (car alist)))
      (car alist)
      (assoc key (cdr alist)))))

(define (assq key alist)
  (if (null? alist)
    #f
    (if (eq? key (car (car alist)))
      (car alist)
      (assq key (cdr alist)))))

(define (member x lst)
  (if (null? lst)
    #f
    (if (equal? x (car lst))
      lst
      (member x (cdr lst)))))

(define (memq x lst)
  (if (null? lst)
    #f
    (if (eq? x (car lst))
      lst
      (memq x (cdr lst)))))

(define (cadr x) (car (cdr x)))
(define (caddr x) (car (cdr (cdr x))))
(define (cdar x) (cdr (car x)))
(define (cddr x) (cdr (cdr x)))
(define (caar x) (car (car x)))
"""
