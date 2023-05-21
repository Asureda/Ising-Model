MODULE mc_routines
use def_var
IMPLICIT NONE
  contains
    subroutine geometry_2D(L,in,nbr)
        IMPLICIT NONE
        integer, intent(in)                         :: L
        integer,dimension(2,L)                      :: in
        integer,dimension(4,N),intent(out)          :: nbr
        integer                                     :: i, y, x

        DO i = 1, L, 1
            in(1,i) = i-1
            in(2,i) = i+1
        END DO
        in(1,1) = L
        in(2,L) = 1

        i = 0
        DO y = 1, L, 1
            DO x=1, L
                i = i+1
                nbr(1,i) = in(2,x) + L*(y-1)
                nbr(2,i) = in(1,x) + L*(y-1)
                nbr(3,i) = x + L*(in(2,y)-1)
                nbr(4,i) = x + L*(in(1,y)-1)
            END DO
        END DO

    end subroutine geometry_2D

    subroutine exp_val( T, d, expval)
      IMPLICIT NONE
      integer, intent(in)                       :: d
      real, intent(in)                          :: T
      real, dimension(-4*d:4*d), intent(out)           :: expval
      integer                                   :: k ,i_min,i_max
      i_min = -4*d
      i_max = 4*d
      DO k = i_min,i_max,2
        expval(k) = exp(-k/T)
        print*,k,expval(k)
      END DO
      return
    end subroutine exp_val

    subroutine update_state(N_spins,d,spin,nbr,r_num1,r_num2)
      IMPLICIT NONE
      integer, intent(in)                         :: N_spins
      real,intent(in)                             ::r_num1,r_num2
      integer                                     :: AE
      integer                                     :: ns, suma ,d
      integer,dimension(N_spins),intent(inout)    :: spin
      integer,dimension(4,N_spins),intent(in)     :: nbr
      real, dimension(-4*d:4*d)                   :: expval


        AE=0
        ns = mod(int(N_spins*r_num1),N_spins) + 1
        suma = spin(nbr(1,ns)) + spin(nbr(2,ns)) + spin(nbr(3,ns)) + spin(nbr(4,ns))
        AE = 2*spin(ns)*suma

        if(AE.lt.0) then
          spin(ns) = -spin(ns)
        else
          if(r_num2.lt.expval(AE)) then
            spin(ns) = -spin(ns)
          end if
        end if

    end subroutine

    subroutine observables(spin,nbr, L, z,E,M,E_accum,M_accum)
        IMPLICIT NONE
        integer                                     :: i, j
        integer                                     :: E_spin
        integer, intent(in)                         :: z, L
        integer,dimension(N_spins)                  :: spin
        integer,dimension(4,N_spins),intent(in)     :: nbr
        integer, intent(inout)                      :: E, M
        integer, intent(out)                        :: E_accum, M_accum



        E_accum = 0.d0 ; M_accum = 0.d0 ;

        DO i = 1, N_spins, 1
            E_spin = 0.d0
            DO j = 1, z, 1
                E_spin = E_spin + spin(nbr(j,i))
            END DO
            E_spin = -spin(i)*E_spin
            E_accum = E_accum + E_spin
            M_accum = M_accum + spin(i)

        END DO
        E_accum = E_accum/2
        M = M + M_accum
        E = E + E_accum
    end subroutine observables

    subroutine print_output(unit_,n_points,E,M)
        implicit none
        integer,intent(in)                             :: unit_,n_points
        integer,dimension(n_points),intent(in)         :: E, M

        DO i = 1 , n_points
          write(unit_,*) i, E(i), M(i)
        END DO

    end subroutine print_output

    function mean(x)

      integer,dimension(:), intent(in):: x
        real*8 ::   mean
        mean=sum(x)/(size(x)-1000)
    end function mean



  END MODULE mc_routines
