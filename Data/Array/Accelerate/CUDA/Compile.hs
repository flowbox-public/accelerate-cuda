{-# LANGUAGE CPP                 #-}
{-# LANGUAGE GADTs               #-}
{-# LANGUAGE PatternGuards       #-}
{-# LANGUAGE RecordWildCards     #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell     #-}
{-# LANGUAGE TupleSections       #-}
-- |
-- Module      : Data.Array.Accelerate.CUDA.Compile
-- Copyright   : [2008..2014] Manuel M T Chakravarty, Gabriele Keller
--               [2009..2014] Trevor L. McDonell
-- License     : BSD3
--
-- Maintainer  : Trevor L. McDonell <tmcdonell@cse.unsw.edu.au>
-- Stability   : experimental
-- Portability : non-portable (GHC extensions)
--

module Data.Array.Accelerate.CUDA.Compile (

  -- * generate and compile kernels to realise a computation
  compileAcc, compileAfun

) where

-- friends
import Data.Array.Accelerate.Error
import Data.Array.Accelerate.Tuple
import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate.CUDA.AST
import Data.Array.Accelerate.CUDA.State
import Data.Array.Accelerate.CUDA.Context
import Data.Array.Accelerate.CUDA.CodeGen
import Data.Array.Accelerate.CUDA.Array.Sugar
import Data.Array.Accelerate.CUDA.Analysis.Launch
import Data.Array.Accelerate.CUDA.Foreign.Import                ( canExecuteAcc, canExecuteExp )
import Data.Array.Accelerate.CUDA.Persistent                    as KT
import qualified Data.Array.Accelerate.CUDA.FullList            as FL
import qualified Data.Array.Accelerate.CUDA.Debug               as D

-- libraries
import Numeric
import Prelude                                                  hiding ( exp, scanl, scanr )
import Control.Applicative                                      hiding ( Const )
import Control.Exception
import Control.Monad
import Control.Monad.Reader                                     ( asks )
import Control.Monad.State                                      ( gets )
import Control.Monad.Trans                                      ( liftIO, MonadIO )
import Control.Concurrent
import Crypto.Hash.MD5                                          ( hashlazy )
import Data.List                                                ( intercalate )
import Data.FileEmbed                                           ( embedDir )
import Data.Maybe
import Data.Monoid
import System.IO.Unsafe
import System.Mem.Weak
import Text.PrettyPrint.Mainland                                ( ppr, renderCompact, displayS )
import qualified Data.ByteString                                as B
import qualified Data.ByteString.Char8                          as BC
import qualified Data.Text.Lazy                                 as T
import qualified Data.Text.Lazy.Encoding                        as T
import qualified Foreign.CUDA.Driver                            as CUDA
import qualified Foreign.CUDA.Analysis                          as CUDA
import qualified Foreign.CUDA.NVRTC.Error                       as CUDA
import qualified Foreign.CUDA.NVRTC.Compile                     as CUDA

#ifdef ACCELERATE_DEBUG
import System.Time
#endif


-- | Initiate code generation, compilation, and data transfer for an array
-- expression. The returned array computation is annotated so to be suitable for
-- execution in the CUDA environment. This includes:
--
--   * list of array variables embedded within scalar expressions
--
--   * kernel object(s) required to executed the kernel
--
compileAcc :: DelayedAcc a -> CIO (ExecAcc a)
compileAcc = compileOpenAcc

compileAfun :: DelayedAfun f -> CIO (ExecAfun f)
compileAfun = compileOpenAfun


compileOpenAfun :: DelayedOpenAfun aenv f -> CIO (PreOpenAfun ExecOpenAcc aenv f)
compileOpenAfun (Alam l)  = Alam  <$> compileOpenAfun l
compileOpenAfun (Abody b) = Abody <$> compileOpenAcc b


compileOpenAcc :: DelayedOpenAcc aenv a -> CIO (ExecOpenAcc aenv a)
compileOpenAcc = traverseAcc
  where
    -- Traverse an open array expression in depth-first order. The top-level
    -- function traverseAcc is intended for manifest arrays that we will
    -- generate CUDA code for. Array valued subterms, which might be manifest or
    -- delayed, are handled separately.
    --
    -- The applicative combinators are used to gloss over that we are passing
    -- around the AST nodes together with a set of free variable indices that
    -- are merged at every step.
    --
    traverseAcc :: forall aenv arrs. DelayedOpenAcc aenv arrs -> CIO (ExecOpenAcc aenv arrs)
    traverseAcc Delayed{} = $internalError "compileOpenAcc" "unexpected delayed array"
    traverseAcc topAcc@(Manifest pacc) =
      case pacc of
        -- Environment and control flow
        Avar ix                 -> node $ pure (Avar ix)
        Alet a b                -> node . pure =<< Alet         <$> traverseAcc a <*> traverseAcc b
        Apply f a               -> node =<< liftA2 Apply        <$> travAF f <*> travA a
        Awhile p f a            -> node =<< liftA3 Awhile       <$> travAF p <*> travAF f <*> travA a
        Acond p t e             -> node =<< liftA3 Acond        <$> travE  p <*> travA  t <*> travA e
        Atuple tup              -> node =<< liftA Atuple        <$> travAtup tup
        Aprj ix tup             -> node =<< liftA (Aprj ix)     <$> travA    tup

        -- Foreign
        Aforeign ff afun a      -> node =<< foreignA ff afun a

        -- Array injection
        Unit e                  -> node =<< liftA  Unit         <$> travE e
        Use arrs                -> use (arrays (undefined::arrs)) arrs >> node (pure $ Use arrs)

        -- Index space transforms
        Reshape s a             -> node =<< liftA2 Reshape              <$> travE s <*> travA a
        Replicate slix e a      -> exec =<< liftA2 (Replicate slix)     <$> travE e <*> travA a
        Slice slix a e          -> exec =<< liftA2 (Slice slix)         <$> travA a <*> travE e
        Backpermute e f a       -> exec =<< liftA3 Backpermute          <$> travE e <*> travF f <*> travA a

        -- Producers
        Generate e f            -> exec =<< liftA2 Generate             <$> travE e <*> travF f
        Map f a                 -> exec =<< liftA2 Map                  <$> travF f <*> travA a
        ZipWith f a b           -> exec =<< liftA3 ZipWith              <$> travF f <*> travA a <*> travA b
        Transform e p f a       -> exec =<< liftA4 Transform            <$> travE e <*> travF p <*> travF f <*> travA a

        -- Consumers
        Fold f z a              -> exec =<< liftA3 Fold                 <$> travF f <*> travE z <*> travA a
        Fold1 f a               -> exec =<< liftA2 Fold1                <$> travF f <*> travA a
        FoldSeg f e a s         -> exec =<< liftA4 FoldSeg              <$> travF f <*> travE e <*> travA a <*> travA s
        Fold1Seg f a s          -> exec =<< liftA3 Fold1Seg             <$> travF f <*> travA a <*> travA s
        Scanl f e a             -> exec =<< liftA3 Scanl                <$> travF f <*> travE e <*> travA a
        Scanl' f e a            -> exec =<< liftA3 Scanl'               <$> travF f <*> travE e <*> travA a
        Scanl1 f a              -> exec =<< liftA2 Scanl1               <$> travF f <*> travA a
        Scanr f e a             -> exec =<< liftA3 Scanr                <$> travF f <*> travE e <*> travA a
        Scanr' f e a            -> exec =<< liftA3 Scanr'               <$> travF f <*> travE e <*> travA a
        Scanr1 f a              -> exec =<< liftA2 Scanr1               <$> travF f <*> travA a
        Permute f d g a         -> exec =<< liftA4 Permute              <$> travF f <*> travA d <*> travF g <*> travA a
        Stencil f b a           -> exec =<< liftA2 (flip Stencil b)     <$> travF f <*> travA a
        Stencil2 f b1 a1 b2 a2  -> exec =<< liftA3 stencil2             <$> travF f <*> travA a1 <*> travA a2
          where stencil2 f' a1' a2' = Stencil2 f' b1 a1' b2 a2'

      where
        use :: ArraysR a -> a -> CIO ()
        use ArraysRunit         ()       = return ()
        use ArraysRarray        arr      = useArrayAsync arr Nothing
        use (ArraysRpair r1 r2) (a1, a2) = use r1 a1 >> use r2 a2

        exec :: (Free aenv, PreOpenAcc ExecOpenAcc aenv arrs) -> CIO (ExecOpenAcc aenv arrs)
        exec (aenv, eacc) = do
          let gamma = makeEnvMap aenv
          kernel <- build topAcc gamma
          return $! ExecAcc (fullOfList kernel) gamma eacc

        node :: (Free aenv', PreOpenAcc ExecOpenAcc aenv' arrs') -> CIO (ExecOpenAcc aenv' arrs')
        node = fmap snd . wrap

        wrap :: (Free aenv', PreOpenAcc ExecOpenAcc aenv' arrs') -> CIO (Free aenv', ExecOpenAcc aenv' arrs')
        wrap = return . liftA (ExecAcc noKernel mempty)

        travA :: DelayedOpenAcc aenv a -> CIO (Free aenv, ExecOpenAcc aenv a)
        travA acc = case acc of
          Manifest{}    -> pure                    <$> traverseAcc acc
          Delayed{..}   -> liftA2 (const EmbedAcc) <$> travF indexD <*> travE extentD

        travAF :: DelayedOpenAfun aenv f -> CIO (Free aenv, PreOpenAfun ExecOpenAcc aenv f)
        travAF afun = pure <$> compileOpenAfun afun

        travAtup :: Atuple (DelayedOpenAcc aenv) a -> CIO (Free aenv, Atuple (ExecOpenAcc aenv) a)
        travAtup NilAtup        = return (pure NilAtup)
        travAtup (SnocAtup t a) = liftA2 SnocAtup <$> travAtup t <*> travA a

        travF :: DelayedOpenFun env aenv t -> CIO (Free aenv, PreOpenFun ExecOpenAcc env aenv t)
        travF (Body b)  = liftA Body <$> travE b
        travF (Lam  f)  = liftA Lam  <$> travF f

        noKernel :: FL.FullList () (AccKernel a)
        noKernel =  FL.FL () ($internalError "compile" "no kernel module for this node") FL.Nil

        fullOfList :: [a] -> FL.FullList () a
        fullOfList []       = $internalError "fullList" "empty list"
        fullOfList [x]      = FL.singleton () x
        fullOfList (x:xs)   = FL.cons () x (fullOfList xs)

        -- If it is a foreign call for the CUDA backend, don't bother compiling
        -- the pure version
        --
        foreignA :: (Arrays a, Arrays r, Foreign f)
                 => f a r
                 -> DelayedAfun (a -> r)
                 -> DelayedOpenAcc aenv a
                 -> CIO (Free aenv, PreOpenAcc ExecOpenAcc aenv r)
        foreignA ff afun a = case canExecuteAcc ff of
          Nothing       -> liftA2 (Aforeign ff)          <$> pure <$> compileAfun afun <*> travA a
          Just _        -> liftA  (Aforeign ff err)      <$> travA a
            where
              err = $internalError "compile" "Executing pure version of a CUDA foreign function"

    -- Traverse a scalar expression
    --
    travE :: DelayedOpenExp env aenv e
          -> CIO (Free aenv, PreOpenExp ExecOpenAcc env aenv e)
    travE exp =
      case exp of
        Var ix                  -> return $ pure (Var ix)
        Const c                 -> return $ pure (Const c)
        PrimConst c             -> return $ pure (PrimConst c)
        IndexAny                -> return $ pure IndexAny
        IndexNil                -> return $ pure IndexNil
        Foreign ff f x          -> foreignE ff f x
        --
        Let a b                 -> liftA2 Let                   <$> travE a <*> travE b
        IndexCons t h           -> liftA2 IndexCons             <$> travE t <*> travE h
        IndexHead h             -> liftA  IndexHead             <$> travE h
        IndexTail t             -> liftA  IndexTail             <$> travE t
        IndexSlice slix x s     -> liftA2 (IndexSlice slix)     <$> travE x <*> travE s
        IndexFull slix x s      -> liftA2 (IndexFull slix)      <$> travE x <*> travE s
        ToIndex s i             -> liftA2 ToIndex               <$> travE s <*> travE i
        FromIndex s i           -> liftA2 FromIndex             <$> travE s <*> travE i
        Tuple t                 -> liftA  Tuple                 <$> travT t
        Prj ix e                -> liftA  (Prj ix)              <$> travE e
        Cond p t e              -> liftA3 Cond                  <$> travE p <*> travE t <*> travE e
        While p f x             -> liftA3 While                 <$> travF p <*> travF f <*> travE x
        PrimApp f e             -> liftA  (PrimApp f)           <$> travE e
        Index a e               -> liftA2 Index                 <$> travA a <*> travE e
        LinearIndex a e         -> liftA2 LinearIndex           <$> travA a <*> travE e
        Shape a                 -> liftA  Shape                 <$> travA a
        ShapeSize e             -> liftA  ShapeSize             <$> travE e
        Intersect x y           -> liftA2 Intersect             <$> travE x <*> travE y

      where
        travA :: (Shape sh, Elt e)
              => DelayedOpenAcc aenv (Array sh e)
              -> CIO (Free aenv, ExecOpenAcc aenv (Array sh e))
        travA a = do
          a'    <- traverseAcc a
          return $ (bind a', a')

        travT :: Tuple (DelayedOpenExp env aenv) t
              -> CIO (Free aenv, Tuple (PreOpenExp ExecOpenAcc env aenv) t)
        travT NilTup        = return (pure NilTup)
        travT (SnocTup t e) = liftA2 SnocTup <$> travT t <*> travE e

        travF :: DelayedOpenFun env aenv t -> CIO (Free aenv, PreOpenFun ExecOpenAcc env aenv t)
        travF (Body b)  = liftA Body <$> travE b
        travF (Lam  f)  = liftA Lam  <$> travF f

        foreignE :: (Elt a, Elt b, Foreign f)
                 => f a b
                 -> DelayedFun () (a -> b)
                 -> DelayedOpenExp env aenv a
                 -> CIO (Free aenv, PreOpenExp ExecOpenAcc env aenv b)
        foreignE ff f x = case canExecuteExp ff of
          -- If it's a foreign function that we can generate code from, just
          -- leave it alone. As the pure function is closed, the array
          -- environment needs to be replaced with one of the right type.
          --
          Just _        -> liftA2 (Foreign ff) <$> pure <$> snd <$> travF f <*> travE x

          -- If the foreign function is not intended for this backend, this node
          -- needs to be replaced by a pure accelerate node giving the same
          -- result. Due to the lack of an 'apply' node in the scalar language,
          -- this is done by substitution.
          --
          Nothing       -> travE (apply f x)
            where
              -- Twiddle the environment variables
              --
              apply :: DelayedFun () (a -> b) -> DelayedOpenExp env aenv a -> DelayedOpenExp env aenv b
              apply (Lam (Body b)) e    = Let e $ weakenEA rebuildAcc wAcc $ weakenE wExp b
              apply _ _                 = error "This was a triumph."

              -- As the expression we want to weaken is closed with respect to the array
              -- environment, the index manipulation function becomes a dummy argument.
              --
              wAcc :: Idx () t -> Idx aenv t
              wAcc _                    = error "I'm making a note here:"

              wExp :: Idx ((),a) t -> Idx (env,a) t
              wExp ZeroIdx              = ZeroIdx
              wExp _                    = error "HUGE SUCCESS"

        bind :: (Shape sh, Elt e) => ExecOpenAcc aenv (Array sh e) -> Free aenv
        bind (ExecAcc _ _ (Avar ix)) = freevar ix
        bind _                       = $internalError "bind" "expected array variable"


-- Applicative
-- -----------
--
liftA4 :: Applicative f => (a -> b -> c -> d -> e) -> f a -> f b -> f c -> f d -> f e
liftA4 f a b c d = f <$> a <*> b <*> c <*> d


-- Compilation
-- -----------

-- Generate, compile, and link code to evaluate an array computation. We use
-- 'unsafePerformIO' here to leverage laziness, so that the 'link' function
-- evaluates and blocks on the external compiler only once the compiled object
-- is truly needed.
--
build :: DelayedOpenAcc aenv a -> Gamma aenv -> CIO [AccKernel a]
build acc aenv = do
  dev   <- asks deviceProperties
  mapM (build1 acc) (codegenAcc dev acc aenv)

build1 :: DelayedOpenAcc aenv a -> CUTranslSkel aenv a -> CIO (AccKernel a)
build1 acc code = do
  context       <- asks activeContext
  let dev       =  deviceProperties context
  table         <- gets kernelTable
  (entry,key)   <- compile table dev code
  let (cta,blocks,smem) = launchConfig acc dev occ
      (mdl,fun,occ)     = unsafePerformIO $ do
        m <- link context table key
        f <- CUDA.getFun m entry
        l <- CUDA.requires f CUDA.MaxKernelThreadsPerBlock
        o <- determineOccupancy acc dev f l
        D.when D.dump_cc (stats entry f o)
        return (m,f,o)
  --
  return $ AccKernel entry fun mdl occ cta smem blocks
  where
    stats name fn occ = do
      regs      <- CUDA.requires fn CUDA.NumRegs
      smem      <- CUDA.requires fn CUDA.SharedSizeBytes
      cmem      <- CUDA.requires fn CUDA.ConstSizeBytes
      lmem      <- CUDA.requires fn CUDA.LocalSizeBytes
      let msg1  = "entry function '" ++ name ++ "' used "
                  ++ shows regs " registers, "  ++ shows smem " bytes smem, "
                  ++ shows lmem " bytes lmem, " ++ shows cmem " bytes cmem"
          msg2  = "multiprocessor occupancy " ++ showFFloat (Just 1) (CUDA.occupancy100 occ) "% : "
                  ++ shows (CUDA.activeThreads occ)      " threads over "
                  ++ shows (CUDA.activeWarps occ)        " warps in "
                  ++ shows (CUDA.activeThreadBlocks occ) " blocks"
      --
      -- make sure kernel/stats are printed together. Use 'intercalate' rather
      -- than 'unlines' to avoid a trailing newline.
      --
      message   $ intercalate "\n     ... " [msg1, msg2]


-- Link a compiled binary and update the associated kernel entry in the hash
-- table. This may entail waiting for the external compilation process to
-- complete. If successful, the temporary files are removed.
--
link :: Context -> KernelTable -> KernelKey -> IO CUDA.Module
link context table key =
  let intErr    = $internalError "link" "missing kernel entry"
      ctx       = deviceContext context
      weak_ctx  = weakContext context
  in do
    entry       <- fromMaybe intErr `fmap` KT.lookup context table key
    case entry of
      CompileCall ptx -> do
        message "waiting for nvrtcCompileProgram..."
        code            <- takeMVar ptx
        mdl             <- CUDA.loadData code
        addFinalizer mdl (module_finalizer weak_ctx key mdl)

        -- Update hash tables and stash the binary object into the persistent
        -- cache
        --
        KT.insert table key $! KernelObject code (FL.singleton ctx mdl)
        KT.persist table code key
        return mdl

      -- If we get a real object back, then this will already be in the
      -- persistent cache, since either it was just read in from there, or we
      -- had to generate new code and the link step above has added it.
      --
      KernelObject bin active
        | Just mdl <- FL.lookup ctx active      -> return mdl
        | otherwise                             -> do
            message "re-linking module for current context"
            mdl                 <- CUDA.loadData bin
            addFinalizer mdl (module_finalizer weak_ctx key mdl)
            KT.insert table key $! KernelObject bin (FL.cons ctx mdl active)
            return mdl


-- Generate and compile code for a single open array expression
--
compile :: KernelTable -> CUDA.DeviceProperties -> CUTranslSkel aenv a -> CIO (String, KernelKey)
compile table dev cunit = do
  context       <- asks activeContext
  exists        <- isJust `fmap` liftIO (KT.lookup context table key)
  unless exists $ do
    message     $  unlines [ show key, code ]
    flags       <- compileFlags
    ptx         <- liftIO $ compileKernel code entry flags
    --
    liftIO $ KT.insert table key (CompileCall ptx)
  --
  return (entry, key)
  where
    entry       = show cunit
    key         = (CUDA.computeCapability dev, hashlazy (T.encodeUtf8 $ T.pack code) )
    code        = flip displayS "" . renderCompact $ ppr cunit


-- Determine the appropriate command line flags to pass to the compiler process.
-- This is dependent on the host architecture and device capabilities.
--
compileFlags :: CIO [String]
compileFlags = do
  CUDA.Compute m n <- CUDA.computeCapability `fmap` asks deviceProperties
  let warnings = D.mode D.dump_cc && D.mode D.verbose
      debug    = D.mode D.debug_cc
  return       $ filter (not . null) $
    [ "-arch=compute_" ++ show m ++ show n
    , if warnings then ""   else "--disable-warnings"
    , if debug    then ""   else "-DNDEBUG"
    , if debug    then "-G" else ""
    , "-std=c++11"
    ]

headers :: [(String, String)]
headers = map (\(a,b) -> (a, BC.unpack b)) $ $(embedDir "cubits")

compileKernel :: String -> String -> [String] -> IO (MVar B.ByteString)
compileKernel code funName flags = do
  mvar  <- newEmptyMVar
  _     <- forkIO $ do

    (ptx, ccT) <- do

      ccBegin <- getTime
      prog    <- CUDA.createProgram code funName headers
      CUDA.compileProgram prog flags
        `catch` \(e :: CUDA.NVRTCException) -> do
          putStrLn =<< CUDA.getProgramLog prog
          putStrLn code
          putMVar mvar (throw e)
      ptx <- CUDA.getPTX prog
      CUDA.destroyProgram prog
      ccEnd <- getTime

      return (ptx, diffTime ccBegin ccEnd)
    --
    let msg2  = "nvrtcCompileProgram: " ++ unwords flags
        msg1  = "compile: " ++ D.showFFloatSIBase (Just 3) 1000 ccT "s"

    message $ intercalate "\n     ... " [msg1, msg2]

    putMVar mvar (BC.pack ptx)
  --
  return mvar


-- Debug
-- -----

-- Get the current wall clock time in picoseconds since the epoch
--
{-# INLINE getTime #-}
getTime :: IO Integer
#ifdef ACCELERATE_DEBUG
getTime = do
  TOD sec pico  <- getClockTime
  return        $! pico + sec * 1000000000000
#else
getTime = return 0
#endif

-- Return the difference between the first and second (later) time in seconds
--
{-# INLINE diffTime #-}
diffTime :: Integer -> Integer -> Double
diffTime t1 t2 = fromIntegral (t2 - t1) * 1E-12

-- Return the number of seconds of wall-clock time it took to execute the given
-- action. Makes sure to `deepseq` or otherwise fully evaluate the action before
-- returning from the task, otherwise there is a good chance you'll just pass a
-- suspension out and the elapsed time will be zero.
--
time :: IO a -> IO (a, Double)
{-# NOINLINE time #-}
time p = do
  start <- getTime
  res   <- p
  end   <- getTime
  return $ (res, diffTime start end)


{-# INLINE message #-}
message :: MonadIO m => String -> m ()
message msg = trace msg $ return ()

{-# INLINE trace #-}
trace :: MonadIO m => String -> m a -> m a
trace msg next = D.message D.dump_cc ("cc: " ++ msg) >> next

