//========================================
// �œK�����[�`��
//========================================
#ifndef __GRAVISBELL_I_NN_OPTIMIZER_H__
#define __GRAVISBELL_I_NN_OPTIMIZER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"
#include"../../Common/Guiddef.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** �œK�����[�`����� */
	enum OptimizerType
	{
		OPTIMIZER_TYPE_SGD,			// SGD
		OPTIMIZER_TYPE_MOMENTUM,	// Momentum
		OPTIMIZER_TYPE_ADADGRAD,	// AdaGrad
		OPTIMIZER_TYPE_RMSPROP,		// RMSprop
		OPTIMIZER_TYPE_ADADELTA,	// AdaDelta
		OPTIMIZER_TYPE_ADAM,		// Adam

		OPTIMIZER_TYPE_COUNT
	};

	/** �œK�����[�`�� */
	class IOptimizer
	{
	public:
		/** �R���X�g���N�^ */
		IOptimizer(){}
		/** �f�X�g���N�^ */
		virtual ~IOptimizer(){}

	public:
		//===========================
		// ��{���
		//===========================
		/** �I�v�e�B�}�C�U�̎�ʂ��擾���� */
		virtual OptimizerType GetTypeCode()const = 0;

	public:
		//===========================
		// ����
		//===========================
		/** �p�����[�^���X�V����.
			@param io_lpParamter	�X�V����p�����[�^.
			@param io_lpDParameter	�p�����[�^�̕ω���. */
		virtual ErrorCode UpdateParameter(F32 io_lpParameter[], const F32 i_lpDParameter[]) = 0;
	};

	/** SGD */
	class iOptimizer_SGD : public IOptimizer
	{
	public:
		/** �R���X�g���N�^ */
		iOptimizer_SGD(){}
		/** �f�X�g���N�^ */
		virtual ~iOptimizer_SGD(){}

	public:
		//===========================
		// ��{���
		//===========================
		/** �n�C�p�[�p�����[�^���X�V����
			@param	i_learnCoeff	�w�K�W�� */
		virtual ErrorCode SetHyperParameter(F32 i_learnCoeff) = 0;
	};

	/** Momentum */
	class iOptimizer_Momentum : public IOptimizer
	{
	public:
		/** �R���X�g���N�^ */
		iOptimizer_Momentum(){}
		/** �f�X�g���N�^ */
		virtual ~iOptimizer_Momentum(){}

	public:
		//===========================
		// ��{���
		//===========================
		/** �n�C�p�[�p�����[�^���X�V����
			@param	i_learnCoeff	�w�K�W��
			@param	i_alpha			����. */
		virtual ErrorCode SetHyperParameter(F32 i_learnCoeff, F32 i_alpha) = 0;
	};

	/** AdaDelta */
	class iOptimizer_AdaDelta : public IOptimizer
	{
	public:
		/** �R���X�g���N�^ */
		iOptimizer_AdaDelta(){}
		/** �f�X�g���N�^ */
		virtual ~iOptimizer_AdaDelta(){}

	public:
		//===========================
		// ��{���
		//===========================
		/** �n�C�p�[�p�����[�^���X�V����
			@param	i_rho			������.
			@param	i_epsilon		�⏕�W��. */
		virtual ErrorCode SetHyperParameter(F32 i_rho, F32 i_epsilon) = 0;
	};

	/** Adam */
	class iOptimizer_Adam : public IOptimizer
	{
	public:
		/** �R���X�g���N�^ */
		iOptimizer_Adam(){}
		/** �f�X�g���N�^ */
		virtual ~iOptimizer_Adam(){}

	public:
		//===========================
		// ��{���
		//===========================
		/** �n�C�p�[�p�����[�^���X�V����
			@param	i_learnCoeff	�w�K�W��
			@param	i_alpha			����.
			@param	i_beta1			������.
			@param	i_beta2			������.
			@param	i_epsilon		�⏕�W��. */
		virtual ErrorCode SetHyperParameter(F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon) = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell



#endif	__GRAVISBELL_I_NN_OPTIMIZER_H__
