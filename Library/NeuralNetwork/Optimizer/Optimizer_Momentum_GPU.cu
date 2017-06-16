//===============================================
// �œK�����[�`��(Momentum)
//===============================================
#include"stdafx.h"

#include<vector>

#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h> // need CUDA_VERSION
#include <cudnn.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_Momentum_GPU : public iOptimizer_Momentum
	{
	private:
		U32 m_parameterCount;	/**< �p�����[�^�� */
		F32 m_learnCoeff;		/**< �w�K�W�� */
		F32 m_alpha;				/**< ������ */

		thrust::device_vector<F32> m_lpLastDParameter;	/**< ���O�̍X�V�̍ۂ̃p�����[�^�ω��� */

		cublasHandle_t cublasHandle;

	public:
		/** �R���X�g���N�^ */
		Optimizer_Momentum_GPU(U32 i_parameterCount)
			:	m_parameterCount	(i_parameterCount)
			,	m_learnCoeff		(0.0f)
			,	m_alpha				(0.0f)
		{
			cublasCreate(&cublasHandle);

			this->m_lpLastDParameter.resize(this->m_parameterCount);
		}
		/** �f�X�g���N�^ */
		virtual ~Optimizer_Momentum_GPU()
		{
			cublasDestroy(cublasHandle);
		}

	public:
		//===========================
		// ��{���
		//===========================
		/** �I�v�e�B�}�C�U�̎�ʂ��擾���� */
		OptimizerType GetTypeCode()const
		{
			return OptimizerType::OPTIMIZER_TYPE_MOMENTUM;
		}
		
		/** �n�C�p�[�p�����[�^���X�V����
			@param	i_learnCoeff	�w�K�W��
			@param	i_alpha			����. */
		virtual ErrorCode SetHyperParameter(F32 i_learnCoeff, F32 i_alpha)
		{
			this->m_learnCoeff = i_learnCoeff;
			this->m_alpha = i_alpha;

			return ErrorCode::ERROR_CODE_NONE;
		}


		//===========================
		// ����
		//===========================
		/** �p�����[�^���X�V����.
			@param io_lpParamter	�X�V����p�����[�^.
			@param io_lpDParameter	�p�����[�^�̕ω���. */
		ErrorCode UpdateParameter(F32 io_lpParameter[], const F32 i_lpDParameter[])
		{
			// �ω��ʍX�V
			// ��������K�p
			cublasSscal_v2(
				this->cublasHandle,
				this->m_parameterCount,
				&this->m_alpha,
				thrust::raw_pointer_cast(&this->m_lpLastDParameter[0]),
				1);
			// �ω��ʂ����Z
			cublasSaxpy_v2(
				this->cublasHandle,
				this->m_parameterCount,
				&this->m_learnCoeff,
				i_lpDParameter,
				1,
				thrust::raw_pointer_cast(&this->m_lpLastDParameter[0]),
				1);


			// �p�����[�^�X�V
			F32 alpha = 1;
			cublasSaxpy_v2(
				this->cublasHandle,
				this->m_parameterCount,
				&alpha,
				thrust::raw_pointer_cast(&this->m_lpLastDParameter[0]),
				1,
				io_lpParameter,
				1);


			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** �I�v�e�B�}�C�U���쐬���� */
	iOptimizer_Momentum* CreateOptimizer_Momentum_GPU(U32 i_parameterCount)
	{
		return new Optimizer_Momentum_GPU(i_parameterCount);
	}
	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode UpdateOptimizer_Momentum_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff, F32 i_alpha)
	{
		iOptimizer_Momentum* pOptimizer = dynamic_cast<iOptimizer_Momentum*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_Momentum_GPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_learnCoeff, i_alpha);

		return ErrorCode::ERROR_CODE_NONE;
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell