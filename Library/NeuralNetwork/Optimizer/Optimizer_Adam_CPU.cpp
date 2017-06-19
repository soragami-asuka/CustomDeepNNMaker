//===============================================
// �œK�����[�`��(Adam)
//===============================================
#include"stdafx.h"

#include<vector>

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_Adam_CPU : public iOptimizer_Adam
	{
	private:
		U32 m_parameterCount;	/**< �p�����[�^�� */

		F32	m_alpha;		/**< ����. */
		F32	m_beta1;		/**< ������. */
		F32	m_beta2;		/**< ������. */
		F32	m_epsilon;		/**< �⏕�W��. */

		std::vector<F32> lpParameterM;
		std::vector<F32> lpParameterV;

		F32 m_beta1Pows;	/**< ��1�̊K��l */
		F32 m_beta2Pows;	/**< ��2�̊K��l */

	public:
		/** �R���X�g���N�^ */
		Optimizer_Adam_CPU(U32 i_parameterCount)
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
		virtual ~Optimizer_Adam_CPU()
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

			if(m_beta1Pows < 0.0f)
				m_beta1Pows = m_beta1;
			if(m_beta2Pows < 0.0f)
				m_beta2Pows = m_beta2;


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

			for(U32 paramNum=0; paramNum<this->m_parameterCount; paramNum++)
			{
				this->lpParameterM[paramNum] = this->m_beta1 * this->lpParameterM[paramNum] + (1.0f - this->m_beta1) * i_lpDParameter[paramNum];
				this->lpParameterV[paramNum] = this->m_beta2 * this->lpParameterV[paramNum] + (1.0f - this->m_beta2) * i_lpDParameter[paramNum] * i_lpDParameter[paramNum];

				F32 tmpM = this->lpParameterM[paramNum] / (1.0f - this->m_beta1Pows);
				F32 tmpV = this->lpParameterV[paramNum] / (1.0f - this->m_beta2Pows);

				io_lpParameter[paramNum] += this->m_alpha * (tmpM / (sqrt(tmpV) + this->m_epsilon));
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** �I�v�e�B�}�C�U���쐬���� */
	iOptimizer_Adam* CreateOptimizer_Adam_CPU(U32 i_parameterCount)
	{
		return new Optimizer_Adam_CPU(i_parameterCount);
	}
	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode UpdateOptimizer_Adam_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon)
	{
		iOptimizer_Adam* pOptimizer = dynamic_cast<iOptimizer_Adam*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_Adam_CPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_alpha, i_beta1, i_beta2, i_epsilon);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell