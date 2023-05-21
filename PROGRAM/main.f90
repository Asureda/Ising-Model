PROGRAM METROPOLIS_ISING
    ! Alexandre Sureda Croguennoc 20/12/2020
use mtmod
use def_var
use mc_routines

! OPEN THE FILE AND READ DATA FROM INPUT
    call get_command_argument(1, fName, status=fStat)
    if (fStat /= 0) then
            print*, 'Any file given ---> Exitting program'
            call exit()
    end if
    file_unit = 11
    open(unit=file_unit, file=trim(fName), status='old')
    call read_input(file_unit, L, d, T,  nMCS, nMeas, seed, N_spins)
    print*,'L',L
    close(file_unit)

    seed = seed + 5*T**2
    print*,'seed',seed
    call sgrnd(seed)


    call cpu_time(start)


    allocate(spin(N_spins),in(2,L),nbr(4,N_spins))
    allocate(expval(-4*d:4*d))
    allocate(E_meas(nMCS/nMeas), M_meas(nMCS/nMeas))
    call init_conf(N_spins,seed,spin)
    call geometry_2D(L,in,nbr)
    call exp_val( T, d, expval)

    do nstep = 1, nMCS
      do istep = 1,N_spins
        AE=0
        ns = mod(int(N_spins*grnd()),N_spins) + 1
        suma = spin(nbr(1,ns)) + spin(nbr(2,ns)) + spin(nbr(3,ns)) + spin(nbr(4,ns))
        AE = 2*spin(ns)*suma

        if(AE.lt.0) then
          spin(ns) = -spin(ns)
        else
          if(grnd().lt.expval(AE)) then
            spin(ns) = -spin(ns)
          end if
        end if
      end do
      if((nstep.ge.1000).and.(mod(nstep,nMeas).eq.0)) then
        measures = measures + 1
      call observables(spin,nbr, L, 4,E,M,E_accum,M_accum)
      E_meas(measures) = E_accum
      M_meas(measures) = M_accum
      end if
    end do

    open(unit = 12, file='results.out')
    call print_output(12,measures,E_meas,M_meas)
    call cpu_time(finish)
    write(*,*) 'E ',mean(E_meas)/N_spins

    write(*,*) 'CPU time: ',finish-start,'seconds'


END PROGRAM METROPOLIS_ISING
