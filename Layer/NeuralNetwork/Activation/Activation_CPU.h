//======================================
// �����֐����C���[
// CPU�����p
//======================================
#ifndef __ACTIVATION_CPU_H__
#define __ACTIVATION_CPU_H__

#include"stdafx.h"

#include"Activation_DATA.hpp"
#include"Activation_FUNC.hpp"
#include"Activation_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

class Activation_CPU : public Activation_Base
{
private:
	// �f�[�^�{��
	class Activation_LayerData_CPU& layerData;

	// ���o�̓o�b�t�@
	std::vector<CONST_BATCH_BUFFER_POINTER> m_lppInputBuffer;		/**< ���Z���̓��̓f�[�^ */
	std::vector<BATCH_BUFFER_POINTER>		m_lppOutputBuffer;		/**< �o�b�`�����p�o�̓o�b�t�@ <�o�b�`��> */
	std::vector<BATCH_BUFFER_POINTER>		m_lppDInputBuffer;		/**< ���͌덷�f�[�^ */
	std::vector<CONST_BATCH_BUFFER_POINTER> m_lppDOutputBuffer;		/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */

	// Get�֐����g���Ə������ׂ������ނ̂ňꎞ�ۑ��p. PreCalculate�Œl���i�[.
	U32 inputBufferCount;				/**< ���̓o�b�t�@�� */
	U32 outputBufferCount;				/**< �o�̓o�b�t�@�� */

	std::vector<F32>						lpCalculateSum;	/**< �ꎞ�v�Z�p�̃o�b�t�@[z][y][x]�̃T�C�Y������ */


	// �������֐�
	F32 (Activation_CPU::*func_activation)(F32);
	F32 (Activation_CPU::*func_dactivation)(F32);

public:
	/** �R���X�g���N�^ */
	Activation_CPU(Gravisbell::GUID guid, class Activation_LayerData_CPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
	/** �f�X�g���N�^ */
	virtual ~Activation_CPU();

public:
	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 GetLayerKind()const;

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Initialize(void);


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	ILayerData& GetLayerData();
	const ILayerData& GetLayerData()const;


public:
	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLearn();

	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessCalculate();


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode PreProcessLoop();


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer);

public:
	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer);


protected:
	//================================
	// �������֐�
	//================================
	// lenear�n
	F32 func_activation_lenear(F32 x);
	F32 func_dactivation_lenear(F32 x);

	// sigmoid�n
	F32 func_activation_sigmoid(F32 x);
	F32 func_dactivation_sigmoid(F32 x);

	F32 func_activation_sigmoid_crossEntropy(F32 x);
	F32 func_dactivation_sigmoid_crossEntropy(F32 x);

	// ReLU�n
	F32 func_activation_ReLU(F32 x);
	F32 func_dactivation_ReLU(F32 x);

	// Leaky-ReLU�n
	F32 func_activation_LeakyReLU(F32 x);
	F32 func_dactivation_LeakyReLU(F32 x);

	// tanh�n
	F32 func_activation_tanh(F32 x);
	F32 func_dactivation_tanh(F32 x);

	// SoftMax�n
	F32 func_activation_SoftMax(F32 x);

};


} // Gravisbell;
} // Layer;
} // NeuralNetwork;

#endif