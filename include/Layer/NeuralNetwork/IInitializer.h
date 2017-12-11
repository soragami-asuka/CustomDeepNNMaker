//========================================
// �p�����[�^���������[�`��
//========================================
#ifndef __GRAVISBELL_I_NN_INITIALIZER_H__
#define __GRAVISBELL_I_NN_INITIALIZER_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"
#include"../../Common/Guiddef.h"
#include"../../Common/IODataStruct.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** ���������[�`�� */
	class IInitializer
	{
	public:
		//===========================
		// �R���X�g���N�^/�f�X�g���N�^
		//===========================
		/** �R���X�g���N�^ */
		IInitializer(){}
		/** �f�X�g���N�^ */
		virtual ~IInitializer(){}

	public:
		//===========================
		// �p�����[�^�̒l���擾
		//===========================

		/** �p�����[�^�̒l���擾����.
			@param	i_inputCount	���͐M����.
			@param	i_outputCount	�o�͐M����. */
		virtual F32 GetParameter(U32 i_inputCount, U32 i_outputCount) = 0;
		/** �p�����[�^�̒l���擾����.
			@param	i_inputStruct	���͍\��.
			@param	i_outputStruct	�o�͍\��. */
		virtual F32 GetParameter(const IODataStruct& i_inputStruct, const IODataStruct& i_outputStruct) = 0;
		/** �p�����[�^�̒l���擾����.
			@param	i_inputStruct	���͍\��.
			@param	i_inputCH		���̓`�����l����.
			@param	i_outputStruct	�o�͍\��.
			@param	i_outputCH		�o�̓`�����l����. */
		virtual F32 GetParameter(const Vector3D<S32>& i_inputStruct, U32 i_inputCH, const Vector3D<S32>& i_outputStruct, U32 i_outputCH) = 0;

		/** �p�����[�^�̒l���擾����.
			@param	i_inputCount		���͐M����.
			@param	i_outputCount		�o�͐M����.
			@param	i_parameterCount	�ݒ肷��p�����[�^��.
			@param	o_lpParameter		�ݒ肷��p�����[�^�̔z��. */
		virtual ErrorCode GetParameter(U32 i_inputCount, U32 i_outputCount, U32 i_parameterCount, F32 o_lpParameter[]) = 0;
		/** �p�����[�^�̒l���擾����.
			@param	i_inputStruct		���͍\��.
			@param	i_outputStruct		�o�͍\��.
			@param	i_parameterCount	�ݒ肷��p�����[�^��.
			@param	o_lpParameter		�ݒ肷��p�����[�^�̔z��. */
		virtual ErrorCode GetParameter(const IODataStruct& i_inputStruct, const IODataStruct& i_outputStruct, U32 i_parameterCount, F32 o_lpParameter[]) = 0;
		/** �p�����[�^�̒l���擾����.
			@param	i_inputStruct		���͍\��.
			@param	i_inputCH			���̓`�����l����.
			@param	i_outputStruct		�o�͍\��.
			@param	i_outputCH			�o�̓`�����l����.
			@param	i_parameterCount	�ݒ肷��p�����[�^��.
			@param	o_lpParameter		�ݒ肷��p�����[�^�̔z��. */
		virtual ErrorCode GetParameter(const Vector3D<S32>& i_inputStruct, U32 i_inputCH, const Vector3D<S32>& i_outputStruct, U32 i_outputCH, U32 i_parameterCount, F32 o_lpParameter[]) = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell



#endif	__GRAVISBELL_I_NN_INITIALIZER_H__
