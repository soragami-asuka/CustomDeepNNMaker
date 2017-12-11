//=====================================
// �p�����[�^�������N���X.
// ��{�\��. ���̏ꍇ�͂����h�����Ă����Ɗy�ɍ���
//=====================================
#ifndef __GRAVISBELL_NN_INITIALIZER_BASE_H__
#define __GRAVISBELL_NN_INITIALIZER_BASE_H__

#include"Layer/NeuralNetwork/IInitializer.h"
#include"RandomUtility.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Initializer_base : public IInitializer
	{
	public:
		/** �R���X�g���N�^ */
		Initializer_base();
		/** �f�X�g���N�^1 */
		virtual ~Initializer_base();


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
		virtual F32 GetParameter(const IODataStruct& i_inputStruct, const IODataStruct& i_outputStruct);
		/** �p�����[�^�̒l���擾����.
			@param	i_inputStruct	���͍\��.
			@param	i_inputCH		���̓`�����l����.
			@param	i_outputStruct	�o�͍\��.
			@param	i_outputCH		�o�̓`�����l����. */
		virtual F32 GetParameter(const Vector3D<S32>& i_inputStruct, U32 i_inputCH, const Vector3D<S32>& i_outputStruct, U32 i_outputCH);

		/** �p�����[�^�̒l���擾����.
			@param	i_inputCount		���͐M����.
			@param	i_outputCount		�o�͐M����.
			@param	i_parameterCount	�ݒ肷��p�����[�^��.
			@param	o_lpParameter		�ݒ肷��p�����[�^�̔z��. */
		virtual ErrorCode GetParameter(U32 i_inputCount, U32 i_outputCount, U32 i_parameterCount, F32 o_lpParameter[]);
		/** �p�����[�^�̒l���擾����.
			@param	i_inputStruct		���͍\��.
			@param	i_outputStruct		�o�͍\��.
			@param	i_parameterCount	�ݒ肷��p�����[�^��.
			@param	o_lpParameter		�ݒ肷��p�����[�^�̔z��. */
		virtual ErrorCode GetParameter(const IODataStruct& i_inputStruct, const IODataStruct& i_outputStruct, U32 i_parameterCount, F32 o_lpParameter[]);
		/** �p�����[�^�̒l���擾����.
			@param	i_inputStruct		���͍\��.
			@param	i_inputCH			���̓`�����l����.
			@param	i_outputStruct		�o�͍\��.
			@param	i_outputCH			�o�̓`�����l����.
			@param	i_parameterCount	�ݒ肷��p�����[�^��.
			@param	o_lpParameter		�ݒ肷��p�����[�^�̔z��. */
		virtual ErrorCode GetParameter(const Vector3D<S32>& i_inputStruct, U32 i_inputCH, const Vector3D<S32>& i_outputStruct, U32 i_outputCH, U32 i_parameterCount, F32 o_lpParameter[]);
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif	__GRAVISBELL_NN_INITIALIZER_ZERO_H__