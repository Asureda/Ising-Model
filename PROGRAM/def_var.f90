MODULE def_var
use mtmod

    IMPLICIT NONE
    real                                    :: start, finish,r_num1,r_num2
    integer                                 :: fStat,file_unit,measures
    character(25)                           :: fName
    integer                                 :: L , N_spins, i, j, x, y, E, M,seed,d
    integer                                 :: nstep,istep,ns,AE,E_accum,M_accum
    integer,allocatable,dimension(:)        :: spin
    integer,allocatable,dimension(:,:)      :: in, nbr
    real,dimension(:),allocatable           :: expval
    integer,allocatable,dimension(:)        :: E_meas, M_meas

    contains

      subroutine read_input(file_num, L, d, T,  nMCS, nMeas, seed,N_spins)
          IMPLICIT NONE
          integer,intent(in)                  :: file_num
          real, intent(out)                   :: T
          integer,intent(out)                 :: L , d, nMCS, nMeas, seed,N_spins

            read(file_num,*) L             ! Linear size
            read(file_num,*) d             ! Dimension
            read(file_num,*) T             ! Temperature
            read(file_num,*) nMCS          ! N attempted Spin Flips
            read(file_num,*) nMeas         ! Number of measurments
            read(file_num,*) seed          ! Seed for the rng

            N_spins = L*L
            measures = 0

              return
      end subroutine read_input

      subroutine init_conf(N_spins,seed,spin)
          IMPLICIT NONE
          integer                                     :: i ,j
          integer,intent(in)                          :: N_spins, seed
          integer,dimension(N_spins),intent(out)      :: spin
          real                                        :: r1279

          DO i = 1, N_spins
            spin(i) = 2*mod(int(2*grnd()),2) - 1
          END DO
          return
      end subroutine init_conf




END MODULE def_var
