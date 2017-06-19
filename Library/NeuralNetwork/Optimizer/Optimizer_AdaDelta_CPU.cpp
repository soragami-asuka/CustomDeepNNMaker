//===============================================
// �œK�����[�`��(AdaDelta)
//===============================================
#include"stdafx.h"

#include<vector>

#include"Layer/NeuralNetwork/IOptimizer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_AdaDelta_CPU : public iOptimizer_AdaDelta
	{
	private:
		U32 m_parameterCount;	/**< �p�����[�^�� */

		F32 m_rho;				/**< ������. */
		F32 m_epsilon;			/**< �⏕�W��. */

		std::vector<F32> lpParameterH;
		std::vector<F32> lpParameterS;
		std::vector<F32> lpParameterV;

	public:
		/** �R���X�g���N�^ */
		Optimizer_AdaDelta_CPU(U32 i_parameterCount)
			:	m_parameterCount	(i_parameterCount)
			,	m_rho				(0.0f)
			,	m_epsilon			(0.0f)
		{
			this->lpParameterH.resize(m_parameterCount, 0.0f);
			this->lpParameterS.resize(m_parameterCount, 0.0f);
			this->lpParameterV.resize(m_parameterCount, 0.0f);
		}
		/** �f�X�g���N�^ */
		virtual ~Optimizer_AdaDelta_CPU()
		{
		}

	public:
		//===========================
		// ��{���
		//===========================
		/** �I�v�e�B�}�C�U�̎�ʂ��擾���� */
		OptimizerType GetTypeCode()const
		{
			return OptimizerType::OPTIMIZER_TYPE_ADADELTA;
		}
		
		/** �n�C�p�[�p�����[�^���X�V����
			@param	i_rho			������.
			@param	i_epsilon		�⏕�W��. */
		virtual ErrorCode SetHyperParameter(F32 i_rho, F32 i_epsilon)
		{
			this->m_rho    = i_rho;
			this->m_epsilon = i_epsilon;

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
				// H�X�V
				this->lpParameterH[paramNum] = this->m_rho * this->lpParameterH[paramNum] + (1.0f - this->m_rho) * (i_lpDParameter[paramNum] * i_lpDParameter[paramNum]);

				// V�X�V
				this->lpParameterV[paramNum] = (sqrt(this->lpParameterS[paramNum] + this->m_epsilon)) *i_lpDParameter[paramNum] / (sqrt(this->lpParameterH[paramNum] + this->m_epsilon));

				// S�X�V
				this->lpParameterS[paramNum] = this->m_rho * this->lpParameterS[paramNum] + (1.0f - this->m_rho) * (this->lpParameterV[paramNum] * this->lpParameterV[paramNum]);

				// �d�ݍX�V
				io_lpParameter[paramNum] = io_lpParameter[paramNum] + this->lpParameterV[paramNum];
			}


			return ErrorCode::ERROR_CODE_NONE;
		}
	};

	/** �I�v�e�B�}�C�U���쐬���� */
	iOptimizer_AdaDelta* CreateOptimizer_AdaDelta_CPU(U32 i_parameterCount)
	{
		return new Optimizer_AdaDelta_CPU(i_parameterCount);
	}
	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode UpdateOptimizer_AdaDelta_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount, F32 i_rho, F32 i_epsilon)
	{
		iOptimizer_AdaDelta* pOptimizer = dynamic_cast<iOptimizer_AdaDelta*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = pOptimizer = CreateOptimizer_AdaDelta_CPU(i_parameterCount);
		}

		pOptimizer->SetHyperParameter(i_rho, i_epsilon);

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell