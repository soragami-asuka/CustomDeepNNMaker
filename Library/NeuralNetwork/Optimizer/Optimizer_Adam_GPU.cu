//===============================================
// �œK�����[�`��(Adam)
//===============================================
#include"stdafx.h"

#include<vector>

#include"Layer/NeuralNetwork/IOptimizer.h"

// CUDA�p
#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)

#define BLOCK_SIZE	(32)

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	namespace
	{
		/** �x�N�g���̗v�f���m�̊|���Z. */
		__global__ void cuda_func_updateParameter(F32* io_lpParameter, const F32* i_lpDParameter, const U32 i_bufferSize, F32* io_lpParameterM, F32* io_lpParameterV, F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon, F32 i_beta1Pows, F32 i_beta2Pows)
		{
			const U32 paramNum = blockIdx.x * BLOCK_SIZE + threadIdx.x;
			if(paramNum >= i_bufferSize)	// ���򂷂邪������warp�����Ȃ̂ŁA�������x�ɉe���͂Ȃ��͂�...
				return;

			io_lpParameterM[paramNum] = i_beta1 * io_lpParameterM[paramNum] + (1.0f - i_beta1) * i_lpDParameter[paramNum];
			io_lpParameterV[paramNum] = i_beta2 * io_lpParameterV[paramNum] + (1.0f - i_beta2) * i_lpDParameter[paramNum] * i_lpDParameter[paramNum];

			F32 tmpM = io_lpParameterM[paramNum] / (1.0f - i_beta1Pows);
			F32 tmpV = io_lpParameterV[paramNum] / (1.0f - i_beta2Pows);

			io_lpParameter[paramNum] += i_alpha * (tmpM / (sqrt(tmpV) + i_epsilon));
		}
	}

	class Optimizer_Adam_GPU : public iOptimizer_Adam
	{
	private:
		U32 m_parameterCount;	/**< �p�����[�^�� */

		F32	m_alpha;		/**< ����. */
		F32	m_beta1;		/**< ������. */
		F32	m_beta2;		/**< ������. */
		F32	m_epsilon;		/**< �⏕�W��. */

		thrust::device_vector<F32> lpParameterM;
		thrust::device_vector<F32> lpParameterV;

		F32 m_beta1Pows;	/**< ��1�̊K��l */
		F32 m_beta2Pows;	/**< ��2�̊K��l */

	public:
		/** �R���X�g���N�^ */
		Optimizer_Adam_GPU(U32 i_parameterCount)
			:	m_parameterCount	(i_parameterCount)
			,	m_alpha		(0.0f)
			,	m_beta1		(0.0f)
			,	m_beta2		(0.0f)
			,	m_epsilon	(0.0f)
			,	m_beta2Pows	(1.0f)	/**< ��2�̊K��l */
			,	m_beta1Pows	(1.0f)	/**< ��1�̊K��l */
		{
			this->lpParameterM.resize(this->m_parameterCount);
			this->lpParameterV.resize(this->m_parameterCount);
		}
		/** �f�X�g���N�^ */
		virtual ~Optimizer_Adam_GPU()
		{
		}

	public:
		//===========================
		// ��{���
		//===========================
		/** �I�v�e�B�}�C�U�̎�ʂ��擾���� */
		OptimizerType GetTypeCode()const
		{
			return OptimizerType::OPTIMIZER_TYPE_ADAM;
		}
		
		/** �n�C�p�[�p�����[�^���X�V����
			@param	i_alpha			����.
			@param	i_beta1			������.
			@param	i_beta2			������.
			@param	i_epsilon		�⏕�W��. */
		virtual ErrorCode SetHyperParameter(F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon)
		{
			m_alpha   = i_alpha;
			m_beta1   = i_beta1;
			m_beta2	  = i_beta2;
			m_epsilon = i_epsilon;

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
			this->m_beta1Pows *= this->m_beta1;
			this->m_beta2Pows *= this->m_beta2;
			
			dim3 grid((this->m_parameterCount +(BLOCK_SIZE - 1))/BLOCK_SIZE , 1, 1);
			dim3 block(BLOCK_SIZE, 1, 1);

			cuda_func_updateParameter<<<grid, block>>>(
				io_lpParameter,
				i_lpDParameter,
				this->m_parameterCount,
				thrust::raw_pointer_cast(&this->lpParameterM[0]),
				thrust::raw_pointer_cast(&this->lpParameterV[0]),
				this->m_alpha, this->m_beta1, this->m_beta2, this->m_epsilon,
				this->m_beta1Pows, this->m_beta2Pows);


			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** �I�v�e�B�}�C�U���쐬���� */
	iOptimizer_Adam* CreateOptimizer_Adam_GPU(U32 i_parameterCount)
	{
		return new Optimizer_Adam_GPU(i_parameterCount);
	}
	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode UpdateOptimizer_Adam_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon)
	{
		iOptimizer_Adam* pOptimizer = dynamic_cast<iOptimizer_Adam*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_Adam_GPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_alpha, i_beta1, i_beta2, i_epsilon);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell