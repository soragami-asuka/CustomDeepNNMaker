//===============================================
// �œK�����[�`��(Momentum)
//===============================================
#include"stdafx.h"

#include<vector>

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_Momentum_CPU : public iOptimizer_Momentum
	{
	private:
		U32 m_parameterCount;	/**< �p�����[�^�� */
		F32 m_learnCoeff;		/**< �w�K�W�� */
		F32 m_alpha;				/**< ������ */

		std::vector<F32> m_lpLastDParameter;	/**< ���O�̍X�V�̍ۂ̃p�����[�^�ω��� */

	public:
		/** �R���X�g���N�^ */
		Optimizer_Momentum_CPU(U32 i_parameterCount)
			:	m_parameterCount	(i_parameterCount)
			,	m_learnCoeff		(0.0f)
			,	m_alpha				(0.0f)
		{
			this->m_lpLastDParameter.resize(m_parameterCount, 0.0f);
		}
		/** �f�X�g���N�^ */
		virtual ~Optimizer_Momentum_CPU()
		{
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
			// �ω��ʂ��X�V
			for(U32 paramNum=0; paramNum<this->m_parameterCount; paramNum++)
			{
				this->m_lpLastDParameter[paramNum] = this->m_alpha * this->m_lpLastDParameter[paramNum] + this->m_learnCoeff * i_lpDParameter[paramNum];
			}

			// �p�����[�^�X�V
			for(U32 paramNum=0; paramNum<this->m_parameterCount; paramNum++)
			{
				io_lpParameter[paramNum] += m_lpLastDParameter[paramNum];
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** �I�v�e�B�}�C�U���쐬���� */
	iOptimizer_Momentum* CreateOptimizer_Momentum_CPU(U32 i_parameterCount)
	{
		return new Optimizer_Momentum_CPU(i_parameterCount);
	}
	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode UpdateOptimizer_Momentum_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff, F32 i_alpha)
	{
		iOptimizer_Momentum* pOptimizer = dynamic_cast<iOptimizer_Momentum*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_Momentum_CPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_learnCoeff, i_alpha);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell