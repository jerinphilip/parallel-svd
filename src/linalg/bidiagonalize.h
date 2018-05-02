#include "linalg.h"

template <class _Tensor>
std::tuple<_Tensor, _Tensor, _Tensor> bidiagonalize(_Tensor A) {
    bool row_go, col_go;
    row_go = col_go = true;

    _Tensor Q_t(A.rows, A.rows);
    _Tensor B(A.rows, A.cols);
    _Tensor P(A.cols, A.cols);
    _Tensor x(0, 0);
    _Tensor y(0, 0);
    _Tensor v(0, 0);
    _Tensor H(0, 0);
    _Tensor K(0, 0);
    
    Q_t = identity(Q_t);
    P = identity(P);
    
    /* find length of diagonal and upper diagonal */
    int diag, udiag;
    if (A.rows < A.cols) {
        diag = A.rows;
        udiag = A.rows;
    } else {
        diag = A.cols;
        udiag = A.cols-1;
    }
    
    /* while loop runs till iterations for both rows and columns are exhausted.
       col loop is run length(diag) times and row loop is run length(udiag)
       times. in case of single element, hh multiplication is skipped         */
    int row_iter, col_iter;
    row_iter = col_iter = 0;
    while(row_go || col_go) {
        /* iterations annihilate below diagonal for cols */
        if(row_iter < diag) {
            /* slice x col out of A */
            block col = block(row_iter, A.rows)(row_iter, row_iter+1);
            x = slice(A, col);
            
            /* generate hh matrix based on x */
            if (x.rows > 1) {
                v = reflector(x);
                H = house(v);
                
                /* check if H needs padding */
                if(H.cols != A.rows) {
                    H = id_pad(H, A.rows);
                }
                
                /* multiply H with A from the left */
                A = H*A;
                A = check_zeros(A);
                
                Q_t = H*Q_t;
            }
            
            row_iter++;
        } else {
            row_go = false;
        }
        
        /* iterations annihilate to the right of 1st upper diagonal for rows */
        if(col_iter < udiag) {      
            /* slice y row out of A */
            block row = block(col_iter, col_iter+1)(col_iter+1, A.cols);
            y = slice(A, row);

            if(y.cols > 1) {
                /* generate hh matrix based on y */
                v = reflector(y.transpose());
                K = house(v);
                
                /* check if K needs padding */
                if(K.rows != A.cols) {
                    K = id_pad(K, A.cols);
                }
                
                /* multiply K with A from the right */
                A = A*K;
                A = check_zeros(A);
                
                P = P*K;
            }
            
            col_iter++;
        } else {
            col_go = false;
        }
    }
    B = check_zeros(A);
    
    auto products = std::make_tuple(Q_t, B, P);
    
    return products;
}
