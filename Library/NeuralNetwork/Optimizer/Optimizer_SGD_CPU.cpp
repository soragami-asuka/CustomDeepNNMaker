//===============================================
// �œK�����[�`��(SGD)
//===============================================
#include"stdafx.h"

#include<stdio.h>

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_SGD_CPU : public iOptimizer_SGD
	{
	private:
		U32 m_parameterCount;	/**< �p�����[�^�� */
		F32 m_learnCoeff;	/**< �w�K�W�� */

	public:
		/** �R���X�g���N�^ */
		Optimizer_SGD_CPU(U32 i_parameterCount)
			:	m_parameterCount	(i_parameterCount)
			,	m_learnCoeff		(0.0f)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~Optimizer_SGD_CPU()
		{
		}

	public:
		//===========================
		// ��{���
		//===========================
		/** �I�v�e�B�}�C�U�̎�ʂ��擾���� */
		OptimizerType GetTypeCode()const
		{
			return OptimizerType::OPTIMIZER_TYPE_SGD;
		}
		
		/** �n�C�p�[�p�����[�^���X�V����
			@param	i_learnCoeff	�w�K�W�� */
		ErrorCode SetHyperParameter(F32 i_learnCoeff)
		{
			this->m_learnCoeff = i_learnCoeff;

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
			for(U32 paramNum=0; paramNum<this->m_parameterCount; paramNum++)
			{
				io_lpParameter[paramNum] += this->m_learnCoeff * i_lpDParameter[paramNum];
			}

			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** �I�v�e�B�}�C�U���쐬���� */
	iOptimizer_SGD* CreateOptimizer_SGD_CPU(U32 i_parameterCount)
	{
		return new Optimizer_SGD_CPU(i_parameterCount);
	}
	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode UpdateOptimizer_SGD_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_learnCoeff)
	{
		iOptimizer_SGD* pOptimizer = dynamic_cast<iOptimizer_SGD*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_SGD_CPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_learnCoeff);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell