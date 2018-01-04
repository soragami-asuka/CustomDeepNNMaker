//=======================================
// �P����͂�����NN�̃��C���[
//=======================================
#ifndef __GRAVISBELL_I_NN_CORE_SINGLE_TO_SINGLE_LAYER_H__
#define __GRAVISBELL_I_NN_CORE_SINGLE_TO_SINGLE_LAYER_H__

#include"../IO/ISingleInputLayer.h"
#include"../IO/ISingleOutputLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** ���C���[�x�[�X */
	class INNCoreSingle2SingleLayer : public virtual IO::ISingleInputLayer, public virtual IO::ISingleOutputLayer
	{
	public:
		/** �R���X�g���N�^ */
		INNCoreSingle2SingleLayer() : ISingleInputLayer(), ISingleOutputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~INNCoreSingle2SingleLayer(){}

	public:
		/** ���Z���������s����.
			�����f�o�C�X�ˑ��̃��������n�����
			@param i_lppInputBuffer		���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			@param o_lppOutputBuffer	�o�̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer) = 0;

		/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����. 
			�����f�o�C�X�ˑ��̃��������n�����.
			@param	i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
			@param	i_lppOutputBuffer	�o�̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		virtual ErrorCode CalculateDInput(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;

		/** �w�K���������s����.
			�����f�o�C�X�ˑ��̃��������n�����.
			@param	i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
			@param	i_lppOutputBuffer	�o�̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		virtual ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif