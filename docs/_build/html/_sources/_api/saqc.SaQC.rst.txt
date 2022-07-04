SaQC
====

.. currentmodule:: saqc

.. autoclass:: SaQC
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~SaQC.attrs
      ~SaQC.data
      ~SaQC.flags

   .. rubric:: Methods Summary

   .. autosummary::

      ~SaQC.assignChangePointCluster
      ~SaQC.assignKNNScore
      ~SaQC.assignRegimeAnomaly
      ~SaQC.calculatePolynomialResiduals
      ~SaQC.calculateRollingResiduals
      ~SaQC.clearFlags
      ~SaQC.concatFlags
      ~SaQC.copy
      ~SaQC.copyField
      ~SaQC.correctDrift
      ~SaQC.correctOffset
      ~SaQC.correctRegimeAnomaly
      ~SaQC.dropField
      ~SaQC.fitPolynomial
      ~SaQC.flagByGrubbs
      ~SaQC.flagByStatLowPass
      ~SaQC.flagByStray
      ~SaQC.flagByVariance
      ~SaQC.flagChangePoints
      ~SaQC.flagConstants
      ~SaQC.flagCrossStatistics
      ~SaQC.flagDriftFromNorm
      ~SaQC.flagDriftFromReference
      ~SaQC.flagDummy
      ~SaQC.flagGeneric
      ~SaQC.flagIsolated
      ~SaQC.flagJumps
      ~SaQC.flagMAD
      ~SaQC.flagMVScores
      ~SaQC.flagManual
      ~SaQC.flagMissing
      ~SaQC.flagOffset
      ~SaQC.flagPatternByDTW
      ~SaQC.flagRaise
      ~SaQC.flagRange
      ~SaQC.flagRegimeAnomaly
      ~SaQC.flagUnflagged
      ~SaQC.forceFlags
      ~SaQC.interpolate
      ~SaQC.interpolateByRolling
      ~SaQC.interpolateIndex
      ~SaQC.interpolateInvalid
      ~SaQC.linear
      ~SaQC.plot
      ~SaQC.processGeneric
      ~SaQC.propagateFlags
      ~SaQC.renameField
      ~SaQC.resample
      ~SaQC.roll
      ~SaQC.selectTime
      ~SaQC.shift
      ~SaQC.transferFlags
      ~SaQC.transform

   .. rubric:: Attributes Documentation

   .. autoattribute:: attrs
   .. autoattribute:: data
   .. autoattribute:: flags

   .. rubric:: Methods Documentation

   .. automethod:: assignChangePointCluster
   .. automethod:: assignKNNScore
   .. automethod:: assignRegimeAnomaly
   .. automethod:: calculatePolynomialResiduals
   .. automethod:: calculateRollingResiduals
   .. automethod:: clearFlags
   .. automethod:: concatFlags
   .. automethod:: copy
   .. automethod:: copyField
   .. automethod:: correctDrift
   .. automethod:: correctOffset
   .. automethod:: correctRegimeAnomaly
   .. automethod:: dropField
   .. automethod:: fitPolynomial
   .. automethod:: flagByGrubbs
   .. automethod:: flagByStatLowPass
   .. automethod:: flagByStray
   .. automethod:: flagByVariance
   .. automethod:: flagChangePoints
   .. automethod:: flagConstants
   .. automethod:: flagCrossStatistics
   .. automethod:: flagDriftFromNorm
   .. automethod:: flagDriftFromReference
   .. automethod:: flagDummy
   .. automethod:: flagGeneric
   .. automethod:: flagIsolated
   .. automethod:: flagJumps
   .. automethod:: flagMAD
   .. automethod:: flagMVScores
   .. automethod:: flagManual
   .. automethod:: flagMissing
   .. automethod:: flagOffset
   .. automethod:: flagPatternByDTW
   .. automethod:: flagRaise
   .. automethod:: flagRange
   .. automethod:: flagRegimeAnomaly
   .. automethod:: flagUnflagged
   .. automethod:: forceFlags
   .. automethod:: interpolate
   .. automethod:: interpolateByRolling
   .. automethod:: interpolateIndex
   .. automethod:: interpolateInvalid
   .. automethod:: linear
   .. automethod:: plot
   .. automethod:: processGeneric
   .. automethod:: propagateFlags
   .. automethod:: renameField
   .. automethod:: resample
   .. automethod:: roll
   .. automethod:: selectTime
   .. automethod:: shift
   .. automethod:: transferFlags
   .. automethod:: transform
