//===============================================
// �œK�����[�`��(SGD)
//===============================================
#include"stdafx.h"

#include<stdio.h>

#include"Optimizer_SGD_base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_SGD_CPU : public Optimizer_SGD_base
	{
	public:
		/** �R���X�g���N�^ */
		Optimizer_SGD_CPU(U32 i_parameterCount)
			:	Optimizer_SGD_base	(i_parameterCount)
		{
		}
		/** �f�X�g���N�^ */
		virtual ~Optimizer_SGD_CPU()
		{
		}

	public:
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
	IOptimizer* CreateOptimizer_SGD_CPU(U32 i_parameterCount)
	{
		return new Optimizer_SGD_CPU(i_parameterCount);
	}
	/** �I�v�e�B�}�C�U���o�b�t�@����쐬���� */
	IOptimizer* CreateOptimizerFromBuffer_SGD_CPU(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize)
	{
		return CreateOptimizerFromBuffer_SGD(i_lpBuffer, i_bufferSize, o_useBufferSize, CreateOptimizer_SGD_CPU);
	}
	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode ChangeOptimizer_SGD_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount)
	{
		Optimizer_SGD_CPU* pOptimizer = dynamic_cast<Optimizer_SGD_CPU*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = CreateOptimizer_SGD_CPU(i_parameterCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell